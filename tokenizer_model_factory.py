from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertForTokenClassification
from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from lstm_model import LSTMTagger

class TokenizerModelFactory():
    def makeTokenizerModel(self, modelName, unilabel=True, num_labels=10,root = '', **kwargs):
        if unilabel:
            return self.makeUnilabelModel(modelName, num_labels=num_labels,root = root **kwargs)
        else:
            return self.makeMultilabelModel(modelName, num_labels=num_labels, root = root,**kwargs)


    def makeUnilabelModel(self, modelName, num_labels=10, root = '', **kwargs):
        if modelName == 'distilbert-base-uncased':
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            model = DistilBertForSequenceClassification.from_pretrained(root + "distilbert-base-uncased", num_labels=num_labels, **kwargs)
        if modelName == 'gpt2':
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model = GPT2ForSequenceClassification.from_pretrained(root + "gpt2", num_labels=num_labels, **kwargs)
            model.resize_token_embeddings(len(tokenizer))
            # add padding token
            model.config.pad_token_id = tokenizer('[PAD]').input_ids[0]
        if modelName == 'bertweet':
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
            model = AutoModelForSequenceClassification.from_pretrained(root +"vinai/bertweet-base", num_labels=num_labels, **kwargs)
        if modelName == 'distilroberta-base':
            tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
            model = AutoModelForSequenceClassification.from_pretrained(root +"distilroberta-base", num_labels=num_labels, **kwargs)
        if modelName == 'lstm':
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            model = LSTMCclassifier(128, 64, 2, tokenizer.vocab_size, num_labels)

        return tokenizer, model

    def makeMultilabelModel(self, modelName, num_labels=10,root = '', **kwargs):
        if modelName in ['distilbert-base-uncased','distilbert2/','distilbert3/']:
            print(root)
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            model = DistilBertForTokenClassification.from_pretrained(root + modelName, num_labels=num_labels, **kwargs)
        if modelName == 'bertweet':
            tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
            model = AutoModelForTokenClassification.from_pretrained(root +"vinai/bertweet-base", num_labels=num_labels, **kwargs)
        if modelName == 'distilroberta-base':
            tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', add_prefix_space=True)
            model = AutoModelForTokenClassification.from_pretrained(root + "distilroberta-base", num_labels=num_labels, **kwargs)
        if modelName == 'lstm':
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            model = LSTMTagger(128, 64, 2, tokenizer.vocab_size, num_labels)
        if modelName == 'albert-base-v2':
            tokenizer = AutoTokenizer.from_pretrained('albert-base-v2', add_prefix_space=True)
            model = AutoModelForTokenClassification.from_pretrained(root + "albert-base-v2", num_labels=num_labels, **kwargs)
        if modelName == 'squeezebert/squeezebert-uncased':
            tokenizer = AutoTokenizer.from_pretrained('squeezebert/squeezebert-uncased', add_prefix_space=True)
            model = AutoModelForTokenClassification.from_pretrained(root + "squeezebert/squeezebert-uncased", num_labels=num_labels, **kwargs)
        if modelName == 'xlnet-base-cased':
            tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased', add_prefix_space=True)
            model = AutoModelForTokenClassification.from_pretrained(root + "xlnet-base-cased", num_labels=num_labels, **kwargs)



        return tokenizer, model
