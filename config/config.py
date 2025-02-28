from yaml import safe_load

class Config:
    def __init__(self, config_path: str) -> None:
        # global config
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))
        self.pretrained_model_save = self.config["model"]["pretrained_model_save"]
        self.output_path = self.config["model"]["output_path"]
        self.image_size = int(self.config["model"]["image_size"])
        self.text_max_length = int(self.config["model"]["text_max_length"])
        self.max_time_steps = int(self.config["model"]["max_time_steps"])

        # datasets
        # self.dataset_path = self.config["dataset"]["dataset_path"]
        self.merge_dataset_path = self.config["dataset"]["merge_dataset_path"]

        #unet
        self.device = self.config["unet_train"]["device"]
        self.epochs = int(self.config["unet_train"]["epochs"])
        self.lr = float(self.config["unet_train"]["lr"])
        self.batch_size = int(self.config["unet_train"]["batch_size"])
        # self.save_steps = int(self.config["unet_train"]["save_steps"])
        self.backdoor_style = self.config["unet_train"]["backdoor_style"]

        # image-encoder
        # self.image_encoder_path = self.config["image_encoder"]["pretrained_model_path"]

        # ip-adapter
        # self.adapter_pretrained_model_save = self.config["ip_adapter"]["pretrained_model_path"]

        # backdoor
        self.backdoor_target_path = self.config["backdoor"]["target_image_path"]