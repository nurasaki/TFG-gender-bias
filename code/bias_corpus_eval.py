import torch
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import pandas as pd
import numpy as np


# Import custom funcions
from bias_utils import setup_models, setup_logger



def get_probabilities(input_ids, attention_mask, unmasked_input_ids, 
                      tokenizer, model, device):


    
    # Send everything to device if available (MPS)
    # =============================================================================> MPS vs. CPU
    # Revisar si el model implementa la oipció MPS (RoBERTA, no ho implementa?)
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model = model.to(device)

    
    # =============================================================================> Get Model Probs.
    
    with torch.no_grad():
    # with torch.inference_mode(): 
    # => better than "torch.no_grad():"
    

        # Get model outputs and probabilities
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = softmax(logits, dim=2)
    

    # =============================================================================> [MASK] indices
    
    # Index ok <mask> (ojo només funciona quan hi ha 1 MASK)
    mask_indices = torch.where(input_ids.to("cpu") == tokenizer.mask_token_id)
    
    # Si hi ha més d'una MASK a inputs_TM.input_ids, no funciona el mètode anterior, s'ha d'agrupar per row
    # i obtenir el primer valor

    # La Mida de "row_idx" ha de ser == len(sents_TM)
    df_masks_idx = pd.DataFrame({
        "row_idx": mask_indices[0],
        "mask_idx": mask_indices[1]
    }).groupby("row_idx").first().reset_index()
    
    
    row_idx, mask_idx = df_masks_idx["row_idx"], df_masks_idx["mask_idx"]
    
    
    # =============================================================================> Get Probs
                                      
                                                                                                      
    # Tensor slicing not allowed in MPS?
    # persons_ids = tokenizer(sents_unmasked, **tokenizer_kwargs).input_ids[row_idx, mask_idx]
    persons_ids = unmasked_input_ids[row_idx, mask_idx]
    
    
    return probs.to("cpu")[row_idx, mask_idx, persons_ids].detach().numpy()


def get_associations(df, model, tokenizer, tokenizer_kwargs, batch_size, device, logger):


    # ==================================================================>> Extract Sentences
#     sents = list(df.Sentence)
#     sents_TM = list(df.Sent_TM)
#     sents_TAM = list(df.Sent_TAM)
    
    
    sent, sent_TM, sent_TAM = [list(df[col]) for col in ['Sentence', 'Sent_TM', 'Sent_TAM']]
    # ==================================================================>> Tokenize
    
    
    # Tokenize sentences
    inputs_TM = tokenizer(sent_TM, **tokenizer_kwargs)
    inputs_TAM = tokenizer(sent_TAM, **tokenizer_kwargs)
    
    # Get input_ids and attention_mask
    input_ids_TM = inputs_TM.input_ids
    input_ids_TAM = inputs_TAM.input_ids
    attention_mask_TM = inputs_TM.attention_mask
    attention_mask_TAM = inputs_TAM.attention_mask
        
    input_ids_unmasked = tokenizer(sent, **tokenizer_kwargs).input_ids



    # ==================================================================>> TensorDataset, Sampler, DataLoader
    # TensorDataset ====> Han de ser tensors
    data = TensorDataset(input_ids_TM, attention_mask_TM, 
                         input_ids_TAM, attention_mask_TAM,
                         input_ids_unmasked)

    
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)


    # ==================================================================>> Get Probabilities
    probs_TM = []  # Traget probs
    probs_TAM = [] # Prior probs


    for step, batch in enumerate(dataloader):


        # Each 20 batches show progress
        if step % 20 == 0:  logger.info(f"batch {step}")


        p_TM = get_probabilities(batch[0], batch[1], batch[4], tokenizer, model, device)
        probs_TM = np.append(probs_TM, p_TM)

        p_TAM = get_probabilities(batch[2], batch[3], batch[4], tokenizer, model, device)
        probs_TAM = np.append(probs_TAM, p_TAM)


        # prior_probs_all = np.append(target_probs_all, probs_TM)
        # get_probabilities2(input_ids_TM, attention_mask_TM, input_ids_TAM, attention_mask_TAM, 
        #                    unmasked_input_ids, tokenizer, model, device)

        # if (step > 0) & (step % 30 == 0): break

    associations = np.log(probs_TM/probs_TAM)



    return probs_TM, probs_TAM, associations
    

def eval_bias_corpus(model_ref, device, eval_file, output_file, tokenizer_kwargs, batch_size):
    """
    Evaluate DataFrame 
    * get probabilities and associatiosn
    * save DataFrame with results
    * return DataFrame
       
    """
    # Carregeum el model BERT
    tokenizer, model = setup_models(model_ref)
    logger = setup_logger("EVAL " + model_ref)

    # Carrega el DataFrame
    df = pd.read_csv(eval_file, sep="\t")


    # Definim arguments 

    # tokenizer_kwargs = dict(padding='longest', return_token_type_ids=False, return_tensors="pt")

    kwargs = dict(df = df, model=model, tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs, 
                  batch_size=batch_size, device=device, logger=logger)

    # Obtenim les probabilitats i associacions per cada models
    probs_TM, probs_TAM, associations = get_associations(**kwargs)

    # Afegim probabilitats i associacions al DataFrame
    df["probs_TM"] = probs_TM
    df["probs_TAM"] = probs_TAM
    df["associations"] = associations

    # Guardem el DataFrame amb els resultats obtinguts
    df.to_csv(output_file, sep="\t", index=False)
    print("File created:", output_file)
    
    return df
