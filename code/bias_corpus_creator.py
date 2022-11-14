import pandas as pd
import json
import os

def make_english_row(prof, word, pattern, gender, prof_gender, mask):
    
    """Create BEC-Pro row"""

    # mask = '[MASK]' # BERT    
    # mask = '<mask>' # RoBERTA 


    word = word.capitalize()
    row = []
    
    
    # for words such as 'this man' only get 'man'
    if len(word.split()) == 2:
        person = word.split()[1]
    else:
        person = word
    
    
    # ==================================================> OJO ERROR!!
    # Algunes professions que continguin la paraula relativa a la persona també es substituirà.
    # P.ex: 
    # "My son is a paterson" => reemplaça per "My [MASK] is a pater[MASK]"


    # sentence
    sentence = pattern.format(word, prof)


    # sent_TM = sentence.replace(person, mask)
    # 
    sent_TM = sentence.replace(person, mask, 1)




    # sentence: masked_attribute
    sent_AM = sentence

    for p in prof.split():
        sent_AM = sent_AM.replace(p, mask)
    
    row.append(sentence)
        
    # sentence: masked target
    row.append(sent_TM)
    




    # ==================================================> OJO ERROR!!




        
    row.append(sent_AM)
    # sentence: masked target and attribute
    for p in prof.split():
        sent_TM = sent_TM.replace(p, mask)
    row.append(sent_TM)





    # template
    row.append(pattern.format('<person subject>', '<profession>'))

    # person:
    if len(word.split()) == 2:
        row.append(word.split()[1])
    else:
        row.append(word)

    # gender
    row.append(gender)

    # profession
    row.append(prof)

    # profession's (statistical) gender
    row.append(prof_gender)

    return row

def make_prof_df(prof_list, sentence_patterns, male_words, female_words, lang, prof_gender, mask):
    data = []

    for pattern in sentence_patterns:
        # ['{} is a {}.',
        #  '{} works as a {}.',
        #  '{} applied for the position of {}.',
        #  '{}, the {}, had a good day at work.',
        #  '{} wants to become a {}.']
        
        
        for word in male_words:
            for prof in prof_list:
                gender = 'male'
                if lang == 'EN':
                    row = make_english_row(prof, word, pattern, gender, prof_gender, mask)
                
                data.append(row)
                
        for word in female_words:
            for prof in prof_list:
                gender = 'female'
                if lang == 'EN':
                    row = make_english_row(prof, word, pattern, gender, prof_gender, mask)

                data.append(row)

    if lang == 'EN':
        data = pd.DataFrame(data, columns=['Sentence', 'Sent_TM', 'Sent_AM', 'Sent_TAM',
                                           'Template', 'Person', 'Gender', 'Profession', 'Prof_Gender'])
    else:
        raise Exception("The language was wrong!")

    return data

def create_corpus_tsv(lang, outdir, filename, persons_file, patterns_file, profs_file, mask):
    
    print("Parameters:")
    print("---" * 20)
    print("lang:", lang)
    print("outdir:", outdir)
    print("filename:", filename)
    print("person_file:", persons_file)
    print("patterns_file:", patterns_file)
    print("profs_file:", profs_file)
    print("mask:", mask)

    # Load person words TXT file
    # ------------------------------------------------------------------------------------------------------------------
    male_words = [p.strip().split(", ")[0] for p in open(persons_file)]
    female_words = [p.strip().split(", ")[1] for p in open(persons_file)]
    # male_words = ['he', 'this man', 'my brother', 'my son', 'my husband', 'my boyfriend', 'my father', 'my uncle','my dad']
    # female_words = ['she', 'this woman', 'my sister', 'my daughter', 'my wife', 'my girlfriend', 'my mother','my aunt', 'my mom']
    

    # Load patterns TXT file
    # ------------------------------------------------------------------------------------------------------------------
    patterns = [p.strip() for p in open(patterns_file)]
  

    # Load professios JSON file
    # ------------------------------------------------------------------------------------------------------------------
    with open(profs_file, 'r', encoding='utf-8') as f:
        professions = json.load(f)
    


    # Create corpus DataFrame
    # ------------------------------------------------------------------------------------------------------------------
    corpus = pd.DataFrame()
    for gender in ['male', 'female', 'balanced']:
        df = make_prof_df(professions[gender], patterns, male_words, female_words, lang, gender, mask)

        # corpus = corpus.append(df, ignore_index=True)
        # Deprecated since version 1.4.0: Use concat() instead.

        corpus = pd.concat([corpus, df])
    
    
    
    
    print('\nThe corpus creation was successful!')
    print("---" * 20)
    print(f"Person words:      | {len(male_words) + len(male_words)} \
        ({len(male_words)} male, {len(male_words)} female)")
    print(f"Sentence patterns: | {len(patterns)}")
    print(f'Professions        | {len(corpus.Profession.unique())}')
    print("---" * 20)

    
    
    print('\nThe corpus has a length of {} sentences and {} columns'.format(len(corpus), len(corpus.columns)))


    # Save corpus in TSV format
    # ------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, filename)
    print("File created:", fullname)

    corpus.to_csv(fullname, sep='\t', index=False)
          
          
    return corpus

