import numpy as np


def encrypt_data(self, data):
    return data


def encrypt_file(file_path):
    pass


def decrypt_file(file_path):
    pass


def load_game_file(game_file_path):
    game = np.load(game_file_path, allow_pickle=True)
    return game


def load_and_decrypt_data(combined_data):
    panel_decrypted_data = combined_data["panel_data"]
    target_decrypted_data = combined_data["target_data"]
    if 'rules_data' in combined_data:
        rules_data = combined_data["rules_data"]
    else:
        rules_data = None
    return panel_decrypted_data, target_decrypted_data, rules_data


def save_game_file(file_path, data_dict):
    np.save(file_path, np.array(data_dict))

