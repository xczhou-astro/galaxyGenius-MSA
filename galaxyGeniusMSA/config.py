import os
import numpy as np
from tomlkit import parse, dumps
import shutil
from .utils import galaxygenius_data_dir, assign_unit

class Configuration:
    
    def __init__(self):
        """
        The configuration class loads and manages configuration files for galaxyGenius-MSA. 
        It provides methods to read TOML configuration files, load main and MSA-specific settings, 
        merge configurations, convert to saveable types, and save configurations to disk. 
        """
        self.dataDir = galaxygenius_data_dir()
        self.main_config_template = self.__read_config(
            os.path.join(self.dataDir, 'config/config.toml')
        )
        self.msa_config_template = self.__read_config(
            os.path.join(self.dataDir, 'config/config_MSA.toml')
        )
        
    def __read_config(self, file: str) -> dict:
        
        with open(file, 'r') as f:
            config = parse(f.read())
            
        return config
    
    def __load_config(self):
        
        self.config = self.__read_config('config.toml')
        self.msa_config = self.__read_config('config_MSA.toml')
        
    def __config_to_dict(self):
        
        config = self.config | self.msa_config
        
        return config
    
    def __convert_to_saveable(self, conf):
        
        for key in list(self.config.keys()):
            if isinstance(conf[key], list):
                self.config[key] = conf[key]
            elif isinstance(conf[key], (int, float, np.int32, np.int64, np.float32, np.float64)):
                self.config[key] = conf[key]    
            else:
                self.config[key] = str(conf[key])
                
        for key in list(self.msa_config.keys()):
            if isinstance(conf[key], list):
                self.msa_config[key] = conf[key]
            elif isinstance(conf[key], (int, float, np.int32, np.int64, np.float32, np.float64)):
                self.msa_config[key] = conf[key]
            else:
                self.msa_config[key] = str(conf[key])
            
    
    def save_config(self, conf):
        
        """
        Save the current configuration (conf) into 'config.toml' and 'config_MSA.toml'.
        Converts values in conf to appropriate types for saving, updates self.config and self.msa_config dicts,
        then writes them to disk. This enables preserving configuration settings for future use.
        """
        
        self.__convert_to_saveable(conf)
        
        with open('config.toml', 'w') as f:
            f.write(dumps(self.config))
            
        with open('config_MSA.toml', 'w') as f:
            f.write(dumps(self.msa_config))
            
    def get_config(self) -> dict:
        
        """
        Retrieves a dictionary containing the full configuration by combining the main and MSA config files,
        assigning units where appropriate. This is typically used to access all current settings for 
        pipeline or application use. If necessary, the configuration files are updated or (re)loaded before returning.
        
        Returns
        -------
        dict
            A merged configuration dictionary with units assigned to relevant entries.
        """
        
        self.__update_config()
        config = self.__config_to_dict()
        config_dict = dict(config)
        
        config_dict = assign_unit(config_dict)
        
        return config_dict
    
    def __update_config(self):
        
        if os.path.exists('config.toml') and os.path.exists('config_MSA.toml'):
            self.__load_config()
        else:
            self.config = self.main_config_template
            self.msa_config = self.msa_config_template
            
    def init(self, workspace: str):
        
        shutil.copy(
            self.main_config_template,
            os.path.join(workspace, 'config.toml')
        )
        shutil.copy(
            self.msa_config_template,
            os.path.join(workspace, 'config_MSA.toml')
        )