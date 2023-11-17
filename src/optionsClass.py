class TrainingOptions:
    def __init__(self):
        # PATH options
        self.data_dir = None  # directory gated dataset, required
        self.log_dir = None  # directory to store logs, required
        self.coeff_fpath = None  # file with stored chebychev coefficients, required
        self.depth_flat_world_fpath = None  # path to flat world npz file, optional

        # TRAINING options
        self.model_name = "gated2gated"  # the name of the folder to save the model in
        self.model_type = "multinetwork"  # model structure to use, choices: ["multinetwork", "multioutput"]
        self.depth_model = "packnet"  # depth model to use, choices: ["packnet", "resnet", "packnet_full"]
        self.img_ext = "png"  # image extension to use, choices: ["png", "tiff"]
        self.exp_num = -1  # experiment number
        self.exp_name = "gated2gated"  # the name of the folder to save the model in
        self.exp_metainfo = "Main Experiment"  # additional info regarding experiment
        self.height = 512  # crop height of the image
        self.width = 1024  # crop width of the image
        self.num_bits = 10  # number of bits for gated image intensity
        self.scales = [0, 1, 2, 3]  # scales used in the loss
        self.frame_ids = [0, -1, 1]  # frames to load
        self.pose_model_type = "separate_resnet"  # normal or shared, choices: ["posecnn", "separate_resnet"]
        self.num_layers = 18  # number of resnet layers, choices: [18, 34, 50, 101, 152]
        self.weights_init = "pretrained"  # pretrained or scratch, choices: ["pretrained", "scratch"]
        self.pose_model_input = "pairs"  # how many images the pose network gets, choices: ["pairs", "all"]
        self.depth_normalizer = 150.0  # constant to normalize depth
        self.train_depth_normalizer = False  # train only a single scalar constant
        self.min_depth = 0.1  # minimum depth
        self.max_depth = 100.0  # maximum depth
        self.snr_mask = False  # whether to use SNR based mask for reprojection loss
        self.intensity_mask = False  # whether to use Intensity based mask for reprojection loss
        self.min_snr_val = 0.04  # Minimum SNR value for SNR mask
        self.dataset = "gated"  # dataset to train on, choices: ["gated"]
        self.split = "gated2gated"  # which training split to use, choices: ["gated2gated"]
        self.dropout = 0.5  # dropout rate for packnet
        self.feat_stack = "A"  # whether to use concatenation(A) or Addition (B), choices: ["A", "B"]
        self.num_convs = 4  # number of up/down levels in UNet

        # OPTIMIZATION OPTION
        self.batch_size = 1  # batch size
        self.learning_rate = 1e-4  # learning rate
        self.start_epoch = 0  # start epoch for non-zero starting option for continuing training
        self.num_epochs = 20  # number of epochs
        self.scheduler_step_size = 15  # step size of the scheduler

        # LOADING options
        self.load_weights_folder = None  # name of model to load
        self.models_to_load = ["depth", "pose_encoder", "pose"]  # models to load

        # ABLATION options
        self.no_ssim = False  # if not to use SSIM loss
        self.cycle_loss = False  # if set, cycle loss is used
        self.cycle_weight = 0.1  # cycle loss weight
        self.temporal_loss = False  # if set, temporal reprojection loss is used
        self.temporal_weight = 1.0  # temporal loss weight
        self.sim_gated = False  # whether to generate gated simulation image
        self.disparity_smoothness = 1e-3  # disparity smoothness weight
        self.v1_multiscale = False  # if set, uses monodepth v1 multiscale
        self.disable_automasking = False  # if set, doesn't do auto-masking
        self.avg_reprojection = False  # if set, uses average reprojection loss
        self.infty_hole_mask = False  # if set, uses a masking scheme for infinite depth close to camera
        self.infty_epoch_start = 0  # start epoch to use infinity masks
        self.close_px_fact = 0.995  # factor to select close pixels to the image
        self.infty_hole_thresh = 0.01  # threshold to consider infinity points
        self.use_batchnorm = False  # whether to use batchnorm2D in packnet module
        self.albedo_offset = 0.0  # constant factor to add to albedo
        self.freeze_pose_net = False  # whether to freeze the training for pose network
        self.clip_depth_grad = -1.0  # clip depth gradient to a certain value if value > 0
        self.passive_supervision = False  # supervise learning of passive image with real one
        self.passive_weight = 0.1  # passive supervision loss weight

        # LOGGING options
        self.log_frequency = 250  # number of batches between each tensorboard log
        self.chkpt_frequency = 250  # number of batches between each checkpoint
        self.save_frequency = 1  # number of epochs between each save

        # SYSTEM options
        self.no_cuda = False  # whether to train on cpu
        self.num_workers = 12  # number of dataloader workers
