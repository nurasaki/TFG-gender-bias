import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging
from torch.nn.functional import softmax
import pandas as pd



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

    Downloaded models => ~/.cache/huggingface/hub
    Model refs:
    * bert-base-uncased
    * bert-base-cased
    * roberta-base
    * BSC-TeMU/roberta-base-ca-cased
    * PlanTL-GOB-ES/roberta-base-ca
    * projecte-aina/roberta-base-ca-v2
    * projecte-aina/roberta-base-ca-cased-pos
    * projecte-aina/roberta-base-ca-v2-cased-ner
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


def get_inputs_logits_probs(tokenizer, model, text, device="mps"):
    """Returns: inputs, logits, probs for a given text"""

    # inputs = tokenizer(text, return_tensors="pt", padding='longest')
    # logits = model(**inputs).logits
    # probs = softmax(logits, dim=2)
    

    # input_ids = input_ids.to(device)
    # attention_mask = attention_mask.to(device)
    # model = model.to(device)


    inputs = tokenizer(text, return_tensors="pt", padding='longest').to(device)
    logits = model.to(device)(**inputs).logits
    probs = softmax(logits, dim=2)

    # with torch.no_grad():
    # # with torch.inference_mode(): 
    # # => better than "torch.no_grad():"

    #     # Get model outputs and probabilities
    #     logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    #     probs = softmax(logits, dim=2)

    return inputs, logits.to("cpu"), probs.to("cpu")


def get_mask_indices(inputs, mask_id, nth=0):
    """Get mask indices for each row.
    
    Returns: 
    * row_idx => range(inputs.shape[0])
    * mask_idx => position of mask_id in each sentence    
    """
    
    
    # Get [MASK] index
    # ====================================================================================
    row_idx, mask_idx = torch.where(inputs.input_ids.to('cpu') == mask_id)
    
    # First or Last:
    # nth(0) => first()
    # nth(-1) => last()
    # mask_pos = 0 if use_first_mask else -1 


    df_idx = pd.DataFrame({
        "row_idx": row_idx, 
        "mask_idx": mask_idx}).groupby("row_idx").nth(nth).reset_index()


    row_idx, mask_idx = df_idx["row_idx"].values, df_idx["mask_idx"].values

    
    return row_idx, mask_idx


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
    """Prints topk masked values, for a given masked text.
    
    Output:
    ------------------------------------------------------------
    La meva filla es diu <mask>. ('<mask>' position: 6) 

      #    token_id      prob   word
    =======================================
      1        2405     4.68%   Maria
      2        6147     3.18%   Anna
      3        7400     2.98%   Núria
      4       14164     2.80%   Laia
      5        6183     2.71%   Marta
    """
    
    print("\n")
    print("---"*20)
    (values, indices), input_idx = get_topk(text, tokenizer, model, k)

    for mask_vals, mask_indices, input_idx in zip(values, indices, input_idx):
    
        print(f"{text} ('{tokenizer.mask_token}' position: {input_idx}) \n")
        print('\033[94m\033[1m{:>3}{:>12}{:>10}   {}\033[0m'.format("#", "token_id", "prob", "word"))
        print("==="*13)
        
        
        fmt_str = '{:>3}{:>12}{:>10.2%}  {}'
        i=1
        for val, ind in zip(mask_vals, mask_indices):
            print(fmt_str.format(i, ind.item(), val.item(), tokenizer.decode(ind)))
            i+=1


def uni_tokenize(tokenizer, word, return_value=-1):
    """
    Return token_id if word is uni-token or -1 if is multi-token.
    
    Examples:
    tokenizer.encode("militar"): [0, 3212, 2]) (uni-token) => 3212
    tokenizer.encode("gimnasta"): [0, 12670, 8291, 2] (multi-token) => -1
    """
    
    encoded = tokenizer.encode(word)
    return encoded[1] if len(encoded) == 3 else return_value


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