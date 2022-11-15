import yaml

from lib.singleton_objects import UIBuilder

with open('ui_config.yaml', 'r') as f:
    ui_conf = yaml.safe_load(f)

if __name__ == "__main__":
    UIBuilder.build_and_run()
