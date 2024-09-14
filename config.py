import configparser

class Config:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('Config/config.ini')

        # Global constants loaded from the config file
        self.STREAMING = config['DEFAULT'].getboolean('STREAMING')
        self.AWS_REGION = config['DEFAULT']['AWS_REGION']
        self.MODEL_ID = config['DEFAULT']['MODEL_ID']
        self.IMAGE_GENERATION_MODEL_UD = config['DEFAULT']['IMAGE_GENERATION_MODEL_UD']
        self.EMBEDDING_MULTIMODAL_MODEL_ID = config['DEFAULT']['EMBEDDING_MULTIMODAL_MODEL_ID']
        self.EMBEDDING_TEXT_MODEL_ID = config['DEFAULT']['EMBEDDING_TEXT_MODEL_ID']
        self.IMAGE_PATH = config['DEFAULT']['IMAGE_PATH']
        self.OUTPUT_PATH = config['DEFAULT']['OUTPUT_PATH']
        self.OPENSEARCH_HOST = config['DEFAULT']['OPENSEARCH_HOST']
        self.OPENSEARCH_PORT = config['DEFAULT'].getint('OPENSEARCH_PORT')
        self.MULTIMODAL_INDEX_NAME = config['DEFAULT']['MULTIMODAL_INDEX_NAME']
        self.TEXT_INDEX_NAME = config['DEFAULT']['TEXT_INDEX_NAME']
        self.MAX_EMBEDDING_IMAGE_SIZE = config['DEFAULT'].getint('MAX_EMBEDDING_IMAGE_SIZE')
        self.MAX_EMBEDDING_IMAGE_DIMENSIONS = config['DEFAULT'].getint('MAX_EMBEDDING_IMAGE_DIMENSIONS')
        self.MAX_INFERENCE_IMAGE_SIZE = config['DEFAULT'].getint('MAX_INFERENCE_IMAGE_SIZE')
        self.MAX_INFERENCE_IMAGE_DIMENSIONS = config['DEFAULT'].getint('MAX_INFERENCE_IMAGE_DIMENSIONS')
        self.MAX_CHAT_IMAGE_SIZE = config['DEFAULT'].getint('MAX_CHAT_IMAGE_SIZE')
        self.MAX_CHAT_IMAGE_DIMENSIONS = config['DEFAULT'].getint('MAX_CHAT_IMAGE_DIMENSIONS')
        self.JPEG_SAVE_QUALITY = config['DEFAULT'].getint('JPEG_SAVE_QUALITY')
        self.DEFAULT_IMAGE_WIDTH = config['DEFAULT'].getint('DEFAULT_IMAGE_WIDTH')
        self.DEFAULT_IMAGE_HEIGHT = config['DEFAULT'].getint('DEFAULT_IMAGE_HEIGHT')
        self.AWS_LAMBDA_FUNCTION_NAME = config['DEFAULT']['AWS_LAMBDA_FUNCTION_NAME']
        self.MAX_OUTPUT_LENGTH = config['DEFAULT'].getint('MAX_OUTPUT_LENGTH')
        self.HANDLE_DOCUMENT_TO_TEXT_IN_CODE = config['DEFAULT'].getboolean('HANDLE_DOCUMENT_TO_TEXT_IN_CODE')
        self.HANDLE_IMAGES_IN_DOCUMENTS = config['DEFAULT'].getboolean('HANDLE_IMAGES_IN_DOCUMENTS')
        self.MIN_RETRY_WAIT_TIME = config['DEFAULT'].getint('MIN_RETRY_WAIT_TIME')
        self.MAX_RETRY_WAIT_TIME = config['DEFAULT'].getint('MAX_RETRY_WAIT_TIME')
        self.MAX_RETRIES = config['DEFAULT'].getint('MAX_RETRIES')
        self.MAX_TOKENS = config['DEFAULT'].getint('MAX_TOKENS')
        self.MAX_LOOPS = config['DEFAULT'].getint('MAX_LOOPS')
        self.MAX_WORKERS = config['DEFAULT'].getint('MAX_WORKERS')
        self.MIN_CHUNK_LENGTH = config['DEFAULT'].getint('MIN_CHUNK_LENGTH')
        self.MAX_CHUNK_LENGTH = config['DEFAULT'].getint('MAX_CHUNK_LENGTH')
        self.BIG_MAX_CHUNK_LENGTH = config['DEFAULT'].getint('BIG_MAX_CHUNK_LENGTH')
        self.BIG_MIN_CHUNK_LENGTH = config['DEFAULT'].getint('BIG_MIN_CHUNK_LENGTH')
        self.MAX_SEARCH_RESULTS = config['DEFAULT'].getint('MAX_SEARCH_RESULTS')
        self.MAX_ARCHIVE_RESULTS = config['DEFAULT'].getint('MAX_ARCHIVE_RESULTS')
        self.MAX_ARXIV_RESULTS = config['DEFAULT'].getint('MAX_ARXIV_RESULTS')
        self.MAX_IMAGE_SEARCH_RESULTS = config['DEFAULT'].getint('MAX_IMAGE_SEARCH_RESULTS')
        self.DEFAULT_TEMPERATURE = config['DEFAULT'].getfloat('DEFAULT_TEMPERATURE')
        self.DEFAULT_SYSTEM_PROMPT = config['DEFAULT']['DEFAULT_SYSTEM_PROMPT']
        self.IMAGE_FORMATS = config['DEFAULT']['IMAGE_FORMATS'].split(', ')
        self.DOCUMENT_FORMATS = config['DEFAULT']['DOCUMENT_FORMATS'].split(', ')
        self.SHORT_IMAGE_DESCRIPTION_PROMPT = config['DEFAULT']['SHORT_IMAGE_DESCRIPTION_PROMPT']
        self.DETAILED_IMAGE_DESCRIPTION_PROMPT = config['DEFAULT']['DETAILED_IMAGE_DESCRIPTION_PROMPT']
        self.IMAGE_FILTER_PROMPT = config['DEFAULT']['IMAGE_FILTER_PROMPT']
        self.TOOLS_TIMEOUT = config['DEFAULT'].getint('TOOLS_TIMEOUT')
        self.CONVERSATION_VOICES = config['DEFAULT']['CONVERSATION_VOICES'].split(', ')
