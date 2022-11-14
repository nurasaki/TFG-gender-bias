import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging


def describe_model(model, tokenizer):
    """
    Prints Model/Tokenizer types and MASK_TOKEN

    Example:
    ----------------------------------------
    => describe_model(bert_model, bert_tokenizer)

    Output:

    Describe model:
    ------------------------------------------------------
    Model type => BertForMaskedLM
    Token type => BertTokenizer
    MASK_TOKEN => [MASK]
    MASK_ID    => 103 
    """
    
    print("Describe model:")
    print("---" * 20)
    #print("Describe model:")
    print("Model type =>", model.__class__.__name__)
    print("Token type =>", tokenizer.__class__.__name__)
    
    # Mostrem el token màscara per el model triat
    MASK_TOKEN = tokenizer.mask_token
    MASK_ID = tokenizer.mask_token_id

    print("MASK_TOKEN =>", MASK_TOKEN)
    print("MASK_ID    =>", MASK_ID, "\n")


def setup_models(model_ref):

    """
    Creates Tokenizer and Model, from model_ref.

    => Parameters
    -------------------------------------------------------------------------------------------------
    * model_ref: identificator or path (if downloaded) of the model


    => Example
    -------------------------------------------------------------------------------------------------
    * tokenizer, model = setup_models('downloaded_models/bert-base-uncased') => downloaded model
    * tokenizer, model = setup_models('bert-base-uncased')                   => downloads from transformers repository. 


    -------------------------------------------------------------------------------------------------
    https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer
    https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer

    """


    print("Creating AutoTokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_ref)

    print("Creating AutoModelForMaskedLM.")
    model = AutoModelForMaskedLM.from_pretrained(model_ref)

    describe_model(model, tokenizer)

    return tokenizer, model


def setup_device():

    """"""

    # Apple M1-Pro => MPS (Metal Performance Shader)
    # ----------------------------------------------------------------------------------------------------
    # https://pytorch.org/docs/stable/notes/mps.html


    print("\n=> Setup GPU device:")
    print("---" * 30)



    print(f"PyTorch version: {torch.__version__}")

    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device      
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")


    # =====================================================================> OJOOOO!!!
    # RoBERTA no pot utilitzar MPS????
    # device = torch.device('cpu')
    # =====================================================================> OJOOOO!!!



    # NVIDIA GPU - (Bartl et al., 2020)
    # ----------------------------------------------------------------------------------------------------
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     print('We will use the GPU:', torch.cuda.get_device_name(0))
    # else:
    #     print('No GPU available, using the CPU instead.')
    #     device = torch.device('cpu')

    return torch.device(device)


def setup_logger(logger_ref):


    # create logger
    logger = logging.getLogger(logger_ref)
    logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False # No need to propagate 

    # Create console handler and set level to debug
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Create/Add formatter to ch
    formatter = logging.Formatter('[%(name)s] %(asctime)s - %(levelname)s => %(message)s')
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger

