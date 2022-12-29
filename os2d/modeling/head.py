import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_coder import Os2dBoxCoder, BoxGridGenerator
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import BoxList, cat_boxlist


def build_os2d_head_creator(do_simple_affine, is_cuda, use_inverse_geom_model, feature_map_stride, feature_map_receptive_field):
    aligner = Os2dAlignment(do_simple_affine, is_cuda, use_inverse_geom_model)
    head_creator = Os2dHeadCreator(aligner, feature_map_stride, feature_map_receptive_field)
    return head_creator


def convert_box_coordinates_local_to_global(resampling_grids, default_boxes_xyxy):
    # get box transformations:
    # x_global = (x_2 - x_1) / 2 * x_local + (x_1 + x_2) / 2 = x_A * x_local + x_B
    # y_global = (y_2 - y_1) / 2 * y_local + (y_1 + y_2) / 2 = y_A * y_local + y_B
    box_transforms_x_A = (default_boxes_xyxy.narrow(-1, 2, 1) - default_boxes_xyxy.narrow(-1, 0, 1)) / 2
    box_transforms_x_B = (default_boxes_xyxy.narrow(-1, 2, 1) + default_boxes_xyxy.narrow(-1, 0, 1)) / 2
    box_transforms_y_A = (default_boxes_xyxy.narrow(-1, 3, 1) - default_boxes_xyxy.narrow(-1, 1, 1)) / 2
    box_transforms_y_B = (default_boxes_xyxy.narrow(-1, 3, 1) + default_boxes_xyxy.narrow(-1, 1, 1)) / 2

    resampling_grids_size = [-1] * resampling_grids.dim()
    resampling_grids_size[-2] = resampling_grids.size(-2)
    resampling_grids_size[-3] = resampling_grids.size(-3)
    add_dims = lambda x: x.unsqueeze(-2).unsqueeze(-3).expand(resampling_grids_size)
    # convert to the original coordinates
    b_x_A = add_dims(box_transforms_x_A)
    b_x_B = add_dims(box_transforms_x_B)
    b_y_A = add_dims(box_transforms_y_A)
    b_y_B = add_dims(box_transforms_y_B)
    resampling_grids_x = resampling_grids.narrow(-1, 0, 1) * b_x_A + b_x_B
    resampling_grids_y = resampling_grids.narrow(-1, 1, 1) * b_y_A + b_y_B
    resampling_grids_global = torch.cat([resampling_grids_x, resampling_grids_y], -1)

    return resampling_grids_global


class Os2dAlignment(nn.Module):
    """该类包含所有与变换计算有关的运算。
    如果增加新的变换类型，只需要改变这个类。
    """
    def __init__(self, do_simple_affine, is_cuda, use_inverse_geom_model):
        super(Os2dAlignment, self).__init__()

        self.model_type = "affine" if not do_simple_affine else "simple_affine" # "affine" or "simple_affine"
        self.use_inverse_geom_model = use_inverse_geom_model

        # 创建参数回归网络
        if self.model_type == "affine":
            transform_net_output_dim = 6
        elif self.model_type == "simple_affine":
            transform_net_output_dim = 4
        else:
            raise(RuntimeError("Unknown transformation model \"{0}\"".format(self.model_type)))

        # 所有这些数字在语义上是不同的，但由于模型架构中的细节，被设置为15。
        # 这些数字必须与网络回归转换参数兼容
        # 按照弱对齐代码，我们在这里使用15
        # 所有的尺寸都是（高，宽）的格式
        # NOTE: 严格来说，这段代码应该能在非正方形的网格中工作，但这一点从未被测试过，所以预计会有错误。
        self.out_grid_size = FeatureMapSize(w=15, h=15)
        self.reference_feature_map_size = FeatureMapSize(w=15, h=15)
        self.network_stride = FeatureMapSize(w=1, h=1)
        self.network_receptive_field = FeatureMapSize(w=15, h=15)
        # eg: input_feature_dim： 输入维度：15*15 = 225,  transform_net_output_dim: 输出维度
        self.input_feature_dim = self.reference_feature_map_size.w * self.reference_feature_map_size.h
        self.parameter_regressor = TransformationNet(output_dim=transform_net_output_dim,
                                                     use_cuda=is_cuda,
                                                     normalization='batchnorm', # if not self.use_group_norm else 'groupnorm',
                                                     kernel_sizes=[7, 5],
                                                     channels=[128, 64],
                                                     input_feature_dim=self.input_feature_dim)



    def prepare_transform_parameters_for_grid_sampler(self, transform_parameters):
        """标准化仿射变换模型的功能:
         - 要么全基于简化改造 self self.model_type (defined in self.__init__)
         - use invertion or not based on self.use_inverse_geom_model (defined in self.__init__)
        准备与 apply_affine_transform_to_grid 一起使用的变换参数
        Args:
            transform_parameters (Tensor[float], size = batch_size x num_params x h^A x w^A) - 包含每个图像类对和图像中每个空间位置的变换参数
            这里的批次大小等于图像批次大小 b^A 和类批次大小 b^C 的乘积
            参数 num_params 的数量对于完整仿射变换等于 6，对于简化版本等于 4（仅限平移和缩放）

        Returns:
             transform_parameters (Tensor[float], size = (batch_size * h^A * w^A) x 2, 3) - 包含为 apply_affine_transform_to_grid 准备的转换参数
        """
        num_params = transform_parameters.size(1)
        transform_parameters = transform_parameters.transpose(0, 1)  # num_params x batch_size x image_height x image_width
        transform_parameters = transform_parameters.contiguous().view(num_params, -1) # num_params x -1

        if self.model_type == "affine":
            assert num_params == 6, 'Affine tranformation parameter vector has to be of dimension 6, have {0} instead'.format(num_params)
            transform_parameters = transform_parameters.transpose(0, 1).view(-1, 2, 3) # -1, 2, 3 - shape for apply_affine_transform_to_grid function
        elif self.model_type == "simple_affine":
            assert num_params == 4, 'Simplified affine tranformation parameter vector has to be of dimension 4, have {0} instead'.format(num_params)
            zeros_to_fill_blanks = torch.zeros_like(transform_parameters[0])
            transform_parameters = torch.stack( [transform_parameters[0], zeros_to_fill_blanks, transform_parameters[1],
                                                 zeros_to_fill_blanks, transform_parameters[2], transform_parameters[3]] ,dim=1)
            transform_parameters = transform_parameters.view(-1, 2, 3)
            # -1, 2, 3 - shape for apply_affine_transform_to_grid function
        else:
            raise(RuntimeError("Unknown transformation model \"{0}\"".format(self.model_type)))

        if self.use_inverse_geom_model:
            assert self.model_type in ["affine", "simple_affine"], "Invertion of the transformation is implemented only for the affine transfomations"
            assert transform_parameters.size(-2) == 2 and transform_parameters.size(-1) == 3, "transform_parameters should be of size ? x 2 x 3 to interpret them ass affine matrix, have {0} instead".format(transform_parameters.size())
            grid_batch_size = transform_parameters.size(0)

            # # slow code:
            # lower_row = torch.tensor([0,0,1], device=transform_parameters.device, dtype=transform_parameters.dtype)
            # lower_row = torch.stack([lower_row.unsqueeze(0)] * grid_batch_size, dim=0)
            # faster version
            lower_row = torch.zeros(grid_batch_size, 1, 3, device=transform_parameters.device, dtype=transform_parameters.dtype)
            lower_row[:, :, 2] = 1

            full_matrices = torch.cat( [transform_parameters, lower_row], dim=1 )

            def robust_inverse(batchedTensor):
                try:
                    inv = torch.inverse(batchedTensor)   #求逆矩阵, batchedTensor: [4800,3,3]
                except:
                    n = batchedTensor.size(1)
                    batchedTensor_reg = batchedTensor.clone().contiguous()
                    for i in range(n):
                        batchedTensor_reg[:,i,i] = batchedTensor_reg[:,i,i] + (1e-5)
                    inv = torch.inverse(batchedTensor_reg)
                return inv

            def batched_inverse(batchedTensor):
                """
                A workaround of a bug in the pytorch backend from here:
                https://github.com/pytorch/pytorch/issues/13276
                """
                if batchedTensor.shape[0] >= 256 * 256 - 1:
                    temp = []
                    for t in torch.split(batchedTensor, 256 * 256 - 1):
                        temp.append(robust_inverse(t))
                    return torch.cat(temp)
                else:
                    return robust_inverse(batchedTensor)

            inverted = batched_inverse(full_matrices)
            transform_parameters = inverted[:,:2,:]
            transform_parameters = transform_parameters.contiguous()

        return transform_parameters  #eg: [4800,3,3]

    def forward(self, corr_maps):
        """
        Args:
            corr_maps (Tensor[float]): a batch_size x num_features x h^A x w^A 包含每个图像类对和图像中每个空间位置的变换参数的张量
            这里的批次大小 batch_size 等于图像批次大小 b^A 和类别批次大小 b^C 的乘积
            通道数 num_features 应该与创建的特征回归网络兼容 (equals 225 for the weakalign models).

        Returns:
            resampling_grids_local_coord (Tensor[float]): a batch_size x h^A x w^A x out_grid_height x out_grid_width x 2 tensor
            张量代表每个批次元素和每个空间位置的计算变换下的点网格。
            Each point has two coordinates: x \in [-1, 1] and y \in  [-1,1].
            警告！每个点都在相应空间位置的局部坐标系中
        """

        batch_size = corr_maps.size(0)
        fm_height = corr_maps.size(-2)
        fm_width = corr_maps.size(-1)
        assert corr_maps.size(1) == self.input_feature_dim, "The dimension 1 of corr_maps={0} should be equal to self.input_feature_dim={1}".format(corr_maps.size(1), self.input_feature_dim)

        # 应用特征回归网络 (initial ReLU + normalization is inside), [1,6,60,80]
        transform_parameters = self.parameter_regressor(corr_maps)  # batch_size x num_params x image_height x image_width

        # 处理变换参数（转换完整的仿射并在需要时求逆）
        transform_parameters = self.prepare_transform_parameters_for_grid_sampler(transform_parameters)

        # 计算变换下网格点的位置
        # this is an analogue of AffineGridGenV2 from
        # https://github.com/ignacio-rocco/weakalign/blob/dd0892af1e05df1765f8a729644a33ed75ee657e/geotnf/transformation.py
        # 请注意，重要的是要有非默认的 align_corners=True，否则结果会有所不同, resampling_grids_local_coord: [4800,15,15,2]
        resampling_grids_local_coord = F.affine_grid(transform_parameters, torch.Size((transform_parameters.size(0), 1, self.out_grid_size.h, self.out_grid_size.w)), align_corners=True)
        # size takes batch_size, num_channels (ignored), grid height, grid width; both height and width are in the range [-1, 1]
        # 坐标在局部框坐标系中

        # 警告！现在我们拥有每个空间位置坐标系中的所有点
        assert resampling_grids_local_coord.ndimension() == 4 and resampling_grids_local_coord.size(-1) == 2 and resampling_grids_local_coord.size(-2) == self.out_grid_size.w and resampling_grids_local_coord.size(-3) == self.out_grid_size.h, "resampling_grids_local_coord should be of size batch_size x out_grid_width x out_grid_height x 2, but have {0}".format(resampling_grids_local_coord.size())
        # [1,60,80,15,15,2],[batch_size, 特征图高，特征图宽，输出网格高，输出网格宽，2]
        resampling_grids_local_coord = resampling_grids_local_coord.view(batch_size, fm_height, fm_width, self.out_grid_size.h, self.out_grid_size.w, 2)

        return resampling_grids_local_coord


def spatial_norm(feature_mask):
    mask_size = feature_mask.size()
    feature_mask = feature_mask.view(mask_size[0], mask_size[1], -1)
    feature_mask = feature_mask / (feature_mask.sum(dim=2, keepdim=True))
    feature_mask = feature_mask.view(mask_size)
    return feature_mask


class Os2dHeadCreator(nn.Module):
    """
    Os2dHeadCreator创建Os2dHead的特定实例，其中包含从lavel目标/查询图像中提取的特征。
    Note: Os2dHeadCreator对象应该是Os2dModel对象的一个子模块，因为它有可训练参数。
        在self.aligner中的TransformNet，Os2dHead对象不应该是Os2dModel的子模块。
        同时，forward方法只在Os2dHead中实现，而不是在Os2dHeadCreator中实现
    """
    def __init__(self, aligner, feature_map_stride, feature_map_receptive_field):
        super(Os2dHeadCreator, self).__init__()
        # 创建对齐模块
        self.aligner = aligner
        # feature_map_receptive_field 和 feature_map_stride： FeatureMapSize(w=16, h=16)， rec_field：感受野大小和步长
        rec_field, stride = self.get_rec_field_and_stride_after_concat_nets(feature_map_receptive_field, feature_map_stride,
                                                                             self.aligner.network_receptive_field, self.aligner.network_stride)
        self.box_grid_generator_image_level = BoxGridGenerator(box_size=rec_field, box_stride=stride)
        self.box_grid_generator_feature_map_level = BoxGridGenerator(box_size=self.aligner.network_receptive_field,
                                                                     box_stride=self.aligner.network_stride)

    @staticmethod
    def get_rec_field_and_stride_after_concat_nets(receptive_field_netA, stride_netA,
                                                   receptive_field_netB, stride_netB):
        """我们将两个网络串联起来  net(x) = netB(netA(x)), 2者的的步长和感受野都是如此。
        该函数计算了组合的步长和感受野。
        """
        if isinstance(receptive_field_netA, FeatureMapSize):
            assert isinstance(stride_netA, FeatureMapSize) and isinstance(receptive_field_netB, FeatureMapSize) and isinstance(stride_netB, FeatureMapSize), "All inputs should be either of type FeatureMapSize or int"
            rec_field_w, stride_w = Os2dHeadCreator.get_rec_field_and_stride_after_concat_nets(receptive_field_netA.w, stride_netA.w,
                                                                                     receptive_field_netB.w, stride_netB.w)
            rec_field_h, stride_h = Os2dHeadCreator.get_rec_field_and_stride_after_concat_nets(receptive_field_netA.h, stride_netA.h,
                                                                                               receptive_field_netB.h, stride_netB.h)
            return FeatureMapSize(w=rec_field_w, h=rec_field_h), FeatureMapSize(w=stride_w, h=stride_h)
        # stride_netA: 16, receptive_field_netB:15, receptive_field_netA: 16, --> 240
        rec_field = stride_netA * (receptive_field_netB - 1) + receptive_field_netA
        stride = stride_netA * stride_netB
        return rec_field, stride

    @staticmethod
    def resize_feature_maps_to_reference_size(ref_size, feature_maps):
        feature_maps_ref_size = []
        for fm in feature_maps:
            assert fm.size(0) == 1, "Can process only batches of size 1, but have {0}".format(fm.size(0))
            num_feature_channels = fm.size(1)

            # resample class features
            identity = torch.tensor([[1, 0, 0], [0, 1, 0]], device=fm.device, dtype=fm.dtype)
            grid_size = torch.Size([1,
                                    num_feature_channels,
                                    ref_size.h,
                                    ref_size.w])  #eg: torch.Size([1, 1024, 15, 15])
            resampling_grid = F.affine_grid(identity.unsqueeze(0), grid_size, align_corners=True)  #eg: [1,15,15,2]
            fm_ref_size = F.grid_sample(fm, resampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            # fm_ref_size: [1,1024,15,15]
            feature_maps_ref_size.append(fm_ref_size)

        feature_maps_ref_size = torch.cat(feature_maps_ref_size, dim=0)
        return feature_maps_ref_size

    def create_os2d_head(self, class_feature_maps):
        # 将所有特征图转换为标准尺寸， reference_feature_map_size： FeatureMapSize(w=15, h=15)
        reference_feature_map_size = self.aligner.reference_feature_map_size
        class_feature_maps_ref_size = self.resize_feature_maps_to_reference_size(reference_feature_map_size, class_feature_maps)
        return Os2dHead(class_feature_maps_ref_size,
                        self.aligner,
                        self.box_grid_generator_image_level,
                        self.box_grid_generator_feature_map_level)


class Os2dHead(nn.Module):
    """此类计算一批输入特征图和一批类别特征图的识别和定位分数。
    类特征映射应该被馈送到构造函数中，该构造函数在内部存储对它们的引用。
    输入特征图应该被馈送到前向方法中。
    此类的实例应该由 Os2dHeadCreator.create_os2d_head 通过传入 class_feature_maps 创建
    """
    def __init__(self, class_feature_maps, aligner,
                       box_grid_generator_image_level,
                       box_grid_generator_feature_map_level,
                       pool_border_width=2):
        super(Os2dHead, self).__init__()

        # initialize class feature maps
        self.class_feature_maps = class_feature_maps   #【1，,1024，,15，,15】
        self.class_batch_size = self.class_feature_maps.size(0)   #eg: 1

        # 类在图像平面中生成与特征图中的位置相对应的box网格
        self.box_grid_generator_image_level = box_grid_generator_image_level
        # 在特征图平面中生成box网格的类
        self.box_grid_generator_feature_map_level = box_grid_generator_feature_map_level

        # 类特征图必须归一化
        self.class_feature_maps = normalize_feature_map_L2(self.class_feature_maps, 1e-5)

        # 为池化激活创建一个mask——现在只需在边缘阻挡几个像素, [1,1,15,15]
        self.class_pool_mask = torch.zeros( (self.class_feature_maps.size(0), 1,
                                             self.class_feature_maps.size(2), self.class_feature_maps.size(3)), # batch_size x 1 x H x W
                                             dtype=torch.float, device=self.class_feature_maps.device)
        self.class_pool_mask[:, :,
                             pool_border_width : self.class_pool_mask.size(-2) - pool_border_width,
                             pool_border_width : self.class_pool_mask.size(-1) - pool_border_width] = 1
        self.class_pool_mask = spatial_norm(self.class_pool_mask)  #[1,1,15,15]

        # create the alignment module
        self.aligner = aligner


    def forward(self, feature_maps):
        """
        Args:
            feature_maps (Tensor[float], size b^A x d x h^A x w^A) - 包含输入图像的特征图
            b^A - batch size
            d - feature dimensionality
            h^A - height of the feature map
            w^A - width of the feature map
​
        Returns:
                # here b^C is the class batch size, i.e., the number of class images contained in self.class_batch_size passed when creating this object
            output_localization (Tensor[float], size b^A x b^C x 4 x h^A x w^A) - the localization output w.r.t. the standard box encoding - computed by DetectionBoxCoder.build_loc_targets
            output_recognition (Tensor[float], size size b^A x b^C x 1 x h^A x w^A) - the recognition output for each of the classes:
                in the [-1, 1] segment, the higher the better match to the class
            output_recognition_transform_detached (Tensor[float], size b^A x b^C x 1 x h^A x w^A) - same as output_recognition,
                but with the computational graph detached from the transformation (for backward  that does not update
                the transofrmation - intended for the negatives)
            corner_coordinates (Tensor[float], size size b^A x b^C x 8 x h^A x w^A) - the corners of the default boxes after
                the transofrmation, datached from the computational graph, for visualisation only
        """
        # get dims
        batch_size = feature_maps.size(0)
        feature_dim = feature_maps.size(1)
        image_fm_size = FeatureMapSize(img=feature_maps)
        class_fm_size = FeatureMapSize(img=self.class_feature_maps)
        feature_dim_for_regression = class_fm_size.h * class_fm_size.w

        class_feature_dim = self.class_feature_maps.size(1)
        assert feature_dim == class_feature_dim, "Feature dimensionality of input={0} and class={1} feature maps has to equal".format(feature_dim, class_feature_dim)

        # L2-normalize the feature map
        feature_maps = normalize_feature_map_L2(feature_maps, 1e-5)

        # 根据类别特征图和查询特征图，获得所有它们之间的相关性， 矩阵运算，得到结果[1,1,15,15,60,80]
        corr_maps = torch.einsum( "bfhw,afxy->abwhxy", self.class_feature_maps, feature_maps )
        # need to try to optimize this with opt_einsum: https://optimized-einsum.readthedocs.io/en/latest/
        # CAUTION: note the switch of dimensions hw to wh. This is done for compatability with the FeatureCorrelation class by Ignacio Rocco https://github.com/ignacio-rocco/ncnet/blob/master/lib/model.py (to be able to load their models)

        # 重塑以具有类似于图像特征图的标准张量的维度相关图, [1,255,60,80]
        corr_maps = corr_maps.contiguous().view(batch_size * self.class_batch_size,
                                                feature_dim_for_regression,
                                                image_fm_size.h,
                                                image_fm_size.w)

        # 计算网格以重新采样相关映射, [1,60,80,15,15,2],[batch_size, 特征图高，特征图宽，输出网格高，输出网格宽，2]
        resampling_grids_local_coord = self.aligner(corr_maps)

        # 根据相关性图，构建分类输出，[1,1,225,60,80],[批次，类别批次，回归特征维度，特征图高，特征图宽]
        cor_maps_for_recognition = corr_maps.contiguous().view(batch_size,
                                                       self.class_batch_size,
                                                       feature_dim_for_regression,
                                                       image_fm_size.h,
                                                       image_fm_size.w)
        # 拆分出类别批次, [批次，类别批次, 特征图高，特征图宽，输出网格高，输出网格宽，2], [1,1,60,80,15,15,2]
        resampling_grids_local_coord = resampling_grids_local_coord.contiguous().view(batch_size,
                                                                                      self.class_batch_size,
                                                                                      image_fm_size.h,
                                                                                      image_fm_size.w,
                                                                                      self.aligner.out_grid_size.h,
                                                                                      self.aligner.out_grid_size.w,
                                                                                      2)
        # 需要重新计算 resampling_grids 到 [-1, 1] 坐标 w.r.t.使用 F.grid_sample 将特征映射到样本点
        # 首先得到参数回归网络的感受野对应的box列表：box sizes为感受野大小，stride为网络步长, default_boxes_xyxy_wrt_fm:[4800,4]
        default_boxes_xyxy_wrt_fm = self.box_grid_generator_feature_map_level.create_strided_boxes_columnfirst(fm_size=image_fm_size)
        # default_boxes_xyxy_wrt_fm: [1,1,60,80,4]
        default_boxes_xyxy_wrt_fm = default_boxes_xyxy_wrt_fm.view(1, 1, image_fm_size.h, image_fm_size.w, 4)
        # 形状意义： 1 (to broadcast to batch_size) x 1 (to broadcast to class batch_size) x  box_grid_height x box_grid_width x 4
        default_boxes_xyxy_wrt_fm = default_boxes_xyxy_wrt_fm.to(resampling_grids_local_coord.device)
        resampling_grids_fm_coord = convert_box_coordinates_local_to_global(resampling_grids_local_coord, default_boxes_xyxy_wrt_fm)

        # covert to coordinates normalized to [-1, 1] (to be compatible with torch.nn.functional.grid_sample), 取最后一维度，取x的坐标，resampling_grids_fm_coord_x:[1,1,60,80,15,15,1]
        resampling_grids_fm_coord_x = resampling_grids_fm_coord.narrow(-1,0,1)
        resampling_grids_fm_coord_y = resampling_grids_fm_coord.narrow(-1,1,1)    # 取最后一维度，取y的坐标，resampling_grids_fm_coord_x:[1,1,60,80,15,15,1]
        # resampling_grids_fm_coord_unit: [1,1,60,80,15,15,2], 归一化一下?
        resampling_grids_fm_coord_unit = torch.cat( [resampling_grids_fm_coord_x / (image_fm_size.w - 1) * 2 - 1,
            resampling_grids_fm_coord_y / (image_fm_size.h - 1) * 2 - 1], dim=-1 )
        # 裁剪以适合图像平面, 值在-1到1之间
        resampling_grids_fm_coord_unit = resampling_grids_fm_coord_unit.clamp(-1, 1)
        # extract and pool matches
        # # slower code:
        # output_recognition = self.resample_of_correlation_map_simple(cor_maps_for_recognition,
        #                                                          resampling_grids_fm_coord_unit,
        #                                                          self.class_pool_mask)

        # 我们使用更快但更晦涩的版本
        output_recognition = self.resample_of_correlation_map_fast(cor_maps_for_recognition,
                                                             resampling_grids_fm_coord_unit,
                                                             self.class_pool_mask)
        if output_recognition.requires_grad:
            output_recognition_transform_detached = self.resample_of_correlation_map_fast(cor_maps_for_recognition,
                                                                                      resampling_grids_fm_coord_unit.detach(),
                                                                                      self.class_pool_mask)
        else:
            # Optimization to make eval faster
            output_recognition_transform_detached = output_recognition

        # build localization targets
        default_boxes_xyxy_wrt_image = self.box_grid_generator_image_level.create_strided_boxes_columnfirst(fm_size=image_fm_size)

        default_boxes_xyxy_wrt_image = default_boxes_xyxy_wrt_image.view(1, 1, image_fm_size.h, image_fm_size.w, 4)
        # 1 (to broadcast to batch_size) x 1 (to broadcast to class batch_size) x  box_grid_height x box_grid_width x 4
        default_boxes_xyxy_wrt_image = default_boxes_xyxy_wrt_image.to(resampling_grids_local_coord.device)
        resampling_grids_image_coord = convert_box_coordinates_local_to_global(resampling_grids_local_coord, default_boxes_xyxy_wrt_image)


        num_pooled_points = self.aligner.out_grid_size.w * self.aligner.out_grid_size.h
        resampling_grids_x = resampling_grids_image_coord.narrow(-1, 0, 1).contiguous().view(-1, num_pooled_points)
        resampling_grids_y = resampling_grids_image_coord.narrow(-1, 1, 1).contiguous().view(-1, num_pooled_points)
        class_boxes_xyxy = torch.stack([resampling_grids_x.min(dim=1)[0],
                                        resampling_grids_y.min(dim=1)[0],
                                        resampling_grids_x.max(dim=1)[0],
                                        resampling_grids_y.max(dim=1)[0]], 1)

        # extract rectangle borders to draw complete boxes
        corner_coordinates = resampling_grids_image_coord[:,:,:,:,[0,-1]][:,:,:,:,:,[0,-1]] # only the corners
        corner_coordinates = corner_coordinates.detach_()
        corner_coordinates = corner_coordinates.view(batch_size, self.class_batch_size, image_fm_size.h, image_fm_size.w, 8) # batch_size x label_batch_size x fm_height x fm_width x 8
        corner_coordinates = corner_coordinates.transpose(3, 4).transpose(2, 3)  # batch_size x label_batch_size x 5 x fm_height x fm_width

        class_boxes = BoxList(class_boxes_xyxy.view(-1, 4), image_fm_size, mode="xyxy")
        default_boxes_wrt_image = BoxList(default_boxes_xyxy_wrt_image.view(-1, 4), image_fm_size, mode="xyxy")
        default_boxes_with_image_batches = cat_boxlist([default_boxes_wrt_image] * batch_size * self.class_batch_size)

        output_localization = Os2dBoxCoder.build_loc_targets(class_boxes, default_boxes_with_image_batches) # num_boxes x 4
        output_localization = output_localization.view(batch_size, self.class_batch_size, image_fm_size.h, image_fm_size.w, 4)  # batch_size x label_batch_size x fm_height x fm_width x 4
        output_localization = output_localization.transpose(3, 4).transpose(2, 3)  # batch_size x label_batch_size x 4 x fm_height x fm_width

        return output_localization, output_recognition, output_recognition_transform_detached, corner_coordinates


    @staticmethod
    def resample_of_correlation_map_fast(corr_maps, resampling_grids_grid_coord, class_pool_mask):
        """该函数根据表示转换网络产生的转换的点网格对相关张量进行重新采样。
        这是 resample_of_correlation_map_simple 的更高效版本
        Args:
            corr_maps (Tensor[float], size=batch_size x class_batch_size x (h^T*w^T) x h^A x w^A):
                该张量包含输入特征与类别特征图之间的相关性。
                此函数重新采样此张量。
                CAUTION: this tensor shows be viewed to batch_size x class_batch_size x w^T x h^T x h^A x w^A (note the switch of w^T and h^T dimensions)
                这恰好能够加载 weakalign repo 的模型
            resampling_grids_grid_coord (Tensor[float], size=batch_size x class_batch_size x h^A x w^A x h^T x w^T x 2):
                该张量包含显示我们需要重新采样的位置的点的非整数坐标
            class_pool_mask (Tensor[float]): size=class_batch_size x 1 x h^T x w^T
                该张量包含mask，在最终平均池化之前将重采样的相关性乘以该mask。
                它掩盖了类特征图的边界特征。

        Returns:
            matches_pooled (Tensor[float]): size=batch_size x class_batch_size x x 1 x h^A x w^A

        时间比较: resample_of_correlation_map_simple 与 resample_of_correlation_map_fast：
            for 2 images, 11 labels, train_patch_width 400, train_patch_height 600 (fm width = 25, fm height = 38)
                CPU time simple: 0.14s
                CPU time fast: 0.11s
                GPU=Geforce GTX 1080Ti
                GPU time simple: 0.010s
                GPU time fast: 0.006s
        """
        # resampling_grids_grid_coord: [1,1,60,80,15,15,2], image_fm_size:FeatureMapSize(w=80, h=60),  template_fm_size: FeatureMapSize(w=15, h=15), corr_maps:[1,1,225,60,60], class_pool_mask:[1,1,15,15]
        batch_size = corr_maps.size(0)
        class_batch_size = corr_maps.size(1)
        template_fm_size = FeatureMapSize(h=resampling_grids_grid_coord.size(-3), w=resampling_grids_grid_coord.size(-2))
        image_fm_size = FeatureMapSize(img=corr_maps)
        assert template_fm_size.w * template_fm_size.h == corr_maps.size(2), 'the number of channels in the correlation map = {0} should match the size of the resampling grid = {1}'.format(corr_maps.size(2), template_fm_size)

        # 内存高效计算将通过将 corr_map 中的 Y 坐标和通道索引合并为一个浮点数来完成
        # 将两个维度合并在一起, corr_map_merged_y_and_id_in_corr_map: [1,1,13500,80]
        corr_map_merged_y_and_id_in_corr_map = corr_maps.contiguous().view(batch_size * class_batch_size,
            1, -1, image_fm_size.w)

        # 注意坐标的奇怪顺序 - 与 Ignacio 网络中的转置坐标有关
        y_grid, x_grid = torch.meshgrid( torch.arange(template_fm_size.h), torch.arange(template_fm_size.w) )
        index_in_corr_map = y_grid + x_grid * template_fm_size.h

        # clamp to strict [-1, 1]
        # convert to torch.double to get more accuracy
        resampling_grids_grid_coord_ = resampling_grids_grid_coord.clamp(-1, 1).to(dtype=torch.double)
        resampling_grids_grid_coord_x_ = resampling_grids_grid_coord_.narrow(-1,0,1)
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_.narrow(-1,1,1)
        # adjust the y coordinate to take into account the index in the corr_map:
        # convert from [-1, 1] to [0, image_fm_size[0]]
        resampling_grids_grid_coord_y_ = (resampling_grids_grid_coord_y_ + 1) / 2 * (image_fm_size.h - 1)
        # merge with the index in corr map [0]
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_.view( [-1] + list(index_in_corr_map.size()) )
        index_in_corr_map = index_in_corr_map.unsqueeze(0)
        index_in_corr_map = index_in_corr_map.to(device=resampling_grids_grid_coord_.device,
                                                 dtype=resampling_grids_grid_coord_.dtype)
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_ + index_in_corr_map * image_fm_size.h
        # convert back to [-1, -1]
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_ / (image_fm_size.h * template_fm_size.h * template_fm_size.w - 1) * 2 - 1
        resampling_grids_grid_coord_y_ = resampling_grids_grid_coord_y_.view_as(resampling_grids_grid_coord_x_)
        resampling_grids_grid_coord_merged_y_and_id_in_corr_map = torch.cat([resampling_grids_grid_coord_x_, resampling_grids_grid_coord_y_], dim=-1)

        # flatten the resampling grid
        resampling_grids_grid_coord_merged_y_and_id_in_corr_map_1d = \
            resampling_grids_grid_coord_merged_y_and_id_in_corr_map.view(batch_size * class_batch_size, -1, 1, 2)
        # extract the required points
        matches_all_channels = F.grid_sample(corr_map_merged_y_and_id_in_corr_map.to(dtype=torch.double),
                                        resampling_grids_grid_coord_merged_y_and_id_in_corr_map_1d,
                                        mode="bilinear", padding_mode='border', align_corners=True)

        matches_all_channels = matches_all_channels.view(batch_size, class_batch_size, 1,
                                                image_fm_size.h * image_fm_size.w,
                                                template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels.to(dtype=torch.float)

        # combine extracted matches using the average pooling w.r.t. the mask of active points defined by class_pool_mask)
        mask = class_pool_mask.view(1, class_batch_size, 1, 1, template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels * mask

        matches_pooled = matches_all_channels.sum(4)
        matches_pooled = matches_pooled.view(batch_size, class_batch_size, 1, image_fm_size.h, image_fm_size.w)
        return matches_pooled

    @staticmethod
    def resample_of_correlation_map_simple(corr_maps, resampling_grids_grid_coord, class_pool_mask):
        """This function resamples the correlation tensor according to the grids of points representing the transformations produces by the transformation network.
        This function is left hear for understanding, use resample_of_correlation_map_fast, which is faster.
        Args:
            corr_maps (Tensor[float], size=batch_size x class_batch_size x (h^T*w^T) x h^A x w^A):
                This tensor contains correlations between of features of the input and class feature maps.
                This function resamples this tensor.
                CAUTION: this tensor shows be viewed to batch_size x class_batch_size x w^T x h^T x h^A x w^A (note the switch of w^T and h^T dimensions)
                This happens to be able to load models of the weakalign repo
            resampling_grids_grid_coord (Tensor[float], size=batch_size x class_batch_size x h^A x w^A x h^T x w^T x 2):
                This tensor contains non-integer coordinates of the points that show where we need to resample
            class_pool_mask (Tensor[float]): size=class_batch_size x 1 x h^T x w^T
                This tensor contains the mask, by which the resampled correlations are multiplied before final average pooling.
                It masks out the border features of the class feature maps.

        Returns:
            matches_pooled (Tensor[float]): size=batch_size x class_batch_size x x 1 x h^A x w^A

        Time comparison resample_of_correlation_map_simple vs resample_of_correlation_map_fast:
            for 2 images, 11 labels, train_patch_width 400, train_patch_height 600 (fm width = 25, fm height = 38)
                CPU time simple: 0.14s
                CPU time fast: 0.11s
                GPU=Geforce GTX 1080Ti
                GPU time simple: 0.010s
                GPU time fast: 0.006s
        """

        batch_size = corr_maps.size(0)
        class_batch_size = corr_maps.size(1)
        template_fm_size = FeatureMapSize(h=resampling_grids_grid_coord.size(-3), w=resampling_grids_grid_coord.size(-2))
        image_fm_size = FeatureMapSize(img=corr_maps)
        assert template_fm_size.w * template_fm_size.h == corr_maps.size(2), 'the number of channels in the correlation map = {0} should match the size of the resampling grid = {1}'.format(corr_maps.size(2), template_fm_size)

        # use a single batch dimension
        corr_maps = corr_maps.view(batch_size * class_batch_size,
                                   corr_maps.size(2),
                                   image_fm_size.h,
                                   image_fm_size.w)
        resampling_grids_grid_coord = resampling_grids_grid_coord.view(batch_size * class_batch_size,
                                                                       image_fm_size.h,
                                                                       image_fm_size.w,
                                                                       template_fm_size.h,
                                                                       template_fm_size.w,
                                                                       2)

        # extract matches from all channels one by one in a loop, and then combine them (using the average pooling w.r.t. the mask of active points defined by class_pool_mask)
        matches_all_channels = []
        # the order of the loops matters
        for template_x in range(template_fm_size.w):
            for template_y in range(template_fm_size.h):
                # note the weird order of coordinates - related to the transposed coordinates in the weakalign network
                channel_id = template_x * template_fm_size.h + template_y

                channel = corr_maps[:,channel_id:channel_id+1,:,:]
                points = resampling_grids_grid_coord[:,:,:,template_y,template_x,:]

                matches_one_channel = F.grid_sample(channel, points, mode="bilinear", padding_mode='border', align_corners=True)
                matches_all_channels.append(matches_one_channel)
        matches_all_channels = torch.stack(matches_all_channels, -1)

        # start pooling: fix all dimensions explicitly mostly to be safe
        matches_all_channels = matches_all_channels.view(batch_size,
                                                         class_batch_size,
                                                         image_fm_size.h,
                                                         image_fm_size.w,
                                                         template_fm_size.h * template_fm_size.w)
        mask = class_pool_mask.view(1, class_batch_size, 1, 1, template_fm_size.h * template_fm_size.w)
        matches_all_channels = matches_all_channels * mask

        matches_pooled = matches_all_channels.sum(4)
        matches_pooled = matches_pooled.view(batch_size, class_batch_size, 1, image_fm_size.h, image_fm_size.w)
        return matches_pooled


def normalize_feature_map_L2(feature_maps, epsilon=1e-6):
    """Note that the code is slightly different from featureL2Norm of
    From https://github.com/ignacio-rocco/ncnet/blob/master/lib/model.py
    """
    return feature_maps / (feature_maps.norm(dim=1, keepdim=True) + epsilon)


class TransformationNet(nn.Module):
    """This class is implemented on top of the FeatureRegression class form the weakalign repo
    https://github.com/ignacio-rocco/weakalign/blob/master/model/cnn_geometric_model.py
    """
    def __init__(self, output_dim=6, use_cuda=True, normalization='batchnorm', kernel_sizes=[7,5], channels=[128,64], input_feature_dim=15*15, num_groups=16):
        super(TransformationNet, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = input_feature_dim
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=k_size//2))
            # Added padding to make this module preserve spatial size

            if normalization.lower() == 'batchnorm':
                nn_modules.append(nn.BatchNorm2d(ch_out))
            elif normalization.lower() == 'groupnorm':
                nn_modules.append(nn.GroupNorm(num_groups, ch_out))

            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        self.linear = nn.Conv2d(ch_out, output_dim, kernel_size=(k_size, k_size), padding=k_size//2)

        # 初始化最后一层，以传递身份转换
        if output_dim==6:
            # assert output_dim==6, "Implemented only for affine transform"
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
            self.linear.bias.data[0] = 1
            self.linear.bias.data[4] = 1
        elif output_dim==4:
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
            self.linear.bias.data[0] = 1
            self.linear.bias.data[2] = 1

        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, corr_maps):
        # normalization
        corr_maps_norm = normalize_feature_map_L2(F.relu(corr_maps))
        # corr_maps_norm = featureL2Norm(F.relu(corr_maps))

        # apply the network
        transform_params = self.linear(self.conv(corr_maps_norm))
        return transform_params

    def freeze_bn(self):
        # Freeze BatchNorm layers
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
