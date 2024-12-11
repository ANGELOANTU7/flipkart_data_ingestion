import json
import os

class ThresholdConfig:
    def __init__(self, config_path: str = "/tmp/threshold_config.json"):
        self.config_path = config_path
        self.adjustment_factors = {
            'blur_factor': 0.5,      # Default: 0.5 (higher = more sensitive to blur)
            'movement_factor': 0.1,   # Default: 0.3 (higher = more sensitive to movement)
            'shake_factor': 1.0       # Default: 1.0 (higher = more sensitive to shake)
        }
        self.load_config()
    
    def load_config(self):
        """Load threshold adjustment factors from config file if it exists"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.adjustment_factors = json.load(f)
    
    def save_config(self):
        """Save current threshold adjustment factors to config file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.adjustment_factors, f, indent=4)
    
    def update_factors(self, blur_factor=None, movement_factor=None, shake_factor=None):
        """Update threshold adjustment factors"""
        if blur_factor is not None:
            self.adjustment_factors['blur_factor'] = blur_factor
        if movement_factor is not None:
            self.adjustment_factors['movement_factor'] = movement_factor
        if shake_factor is not None:
            self.adjustment_factors['shake_factor'] = shake_factor
        self.save_config()
