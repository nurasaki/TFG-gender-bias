import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging
from torch.nn.functional import softmax


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

        

    Downloaded models => /Users/nurasaki/.cache/huggingface/hub
    * bert-base-uncased
    * roberta-base

    * BSC-TeMU/roberta-base-ca-cased
    * PlanTL-GOB-ES/roberta-base-ca
    * projecte-aina/roberta-base-ca-cased-tc
    * projecte-aina/roberta-base-ca-v2

    * projecte-aina/roberta-base-ca-cased-pos
    * projecte-aina/roberta-base-ca-v2-cased-ner


    * downloaded-bert-base-uncased
    * dbmdz/bert-large-cased-finetuned-conll03-english
    * bert-base-cased
    * version.txt


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


def get_topk(text, tokenizer, model, k):


    # Tokenize
    # ==========================================================================================
    tokenizer_kwargs = dict(padding='longest', return_token_type_ids=False, return_tensors="pt")
    inputs = tokenizer(text, **tokenizer_kwargs).to("cpu")
    input_ids = inputs.input_ids

    
    # Get model outputs and probabilities
    # ==========================================================================================
    # logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = model.to("cpu")(**inputs).logits
    probs = softmax(logits, dim=2)
    
    
    # Index ok <mask> (ojo només funciona quan hi ha 1 MASK)
    # ==========================================================================================
    row_idx, mask_idx = torch.where(input_ids.to("cpu") == tokenizer.mask_token_id)

    return probs[row_idx, mask_idx].topk(k), mask_idx



def print_topk(text, tokenizer, model, k):

    
    print("\n")
    print(text)
    print("---"*20)
    
    (values, indices), input_idx = get_topk(text, tokenizer, model, k)
    # values, indices = torch.topk(probs[row_id][mask_id], 100)
    
    for mask_vals, mask_indices, input_idx in zip(values, indices, input_idx):
    
        print(f"TOKEN_MASK, input index {input_idx}:")
        print("==="*10)
        
        for val, ind in zip(mask_vals, mask_indices):
            print(f'{val.item():6.2%} {ind.item():7} {tokenizer.decode(ind)}')



    # for val, ind in zip(values, indices):
    #     print(f'{val.item():6.2%} {ind.item():7} {tokenizer.decode(ind)}')


def describe_association(df_row, tokenizer, model):
    """Show associations in Catalan row
    
    p_TM => target/profession masket
    p_TAM => target/profession masked with attribute/person masked
    """


    sentence = df_row.Sentence
    sent_TM = df_row.Sent_TM
    sent_TAM = df_row.Sent_TAM

    inputs_unmasked = tokenizer(sentence, return_tensors="pt")
    inputs_TM = tokenizer(sent_TM, return_tensors="pt")
    inputs_TAM = tokenizer(sent_TAM, return_tensors="pt")


    # inputs_TM.to(device)
    # inputs_TAM.to(device)

    assert inputs_TM.input_ids.size() == inputs_unmasked.input_ids.size(), \
        "'inputs_TM' not= 'inputs_unmasked': " + sentence 
    assert inputs_TAM.input_ids.size() == inputs_unmasked.input_ids.size(), \
        "'inputs_TAM' not= 'inputs_unmasked': " + sentence

    # with torch.no_grad():
    #     ...

    probs_TM = softmax(model(**inputs_TM).logits, dim=2)
    probs_TAM = softmax(model(**inputs_TAM).logits, dim=2)

    # OJO!! implementar opció quan hi ha més d'una <MASK>

    # mask_indices = torch.where(inputs_TM.input_ids == tokenizer.mask_token_id)
    row_idx, mask_idx = torch.where(inputs_TM.input_ids == tokenizer.mask_token_id)

    # En el nostre cas "professions", en el cas de Bartl "persones"
    masked_ids = inputs_unmasked.input_ids[row_idx, mask_idx]

    p_TM = probs_TM[row_idx, mask_idx, masked_ids].item()
    p_TAM = probs_TAM[row_idx, mask_idx, masked_ids].item()
    assoc = np.log(p_TM/p_TAM)

    fmt_str = "{:<25}{:>12.6f}{:>12.6f}{:>7.3f}"
    print(fmt_str.format(sentence, p_TM, p_TAM, assoc))