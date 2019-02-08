DATA_DIR = 'data\\'
AUDIO_DIR = DATA_DIR + 'audio\\'
FILTERED_AUDIO_DIR = DATA_DIR + 'filtered_audio\\'
AUGMENTED_AUDIO_DIR = DATA_DIR + 'augmented_audio\\'
SPECTROGRAMS_DIR = DATA_DIR + 'spectrograms\\'
MODELS_DIR = 'models\\'

CLASSES = {0: 'air_conditioner', 1: 'car_horn', 2: 'children_playing', 3: 'dog_bark', 4: 'drilling', 5: 'engine_idling',
           6: 'gun_shot', 7: 'jackhammer', 8: 'siren', 9: 'street_music'}
NB_CLASSES = len(CLASSES)