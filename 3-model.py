from paths import WIKIPEDIA_HOME
from paths import LOG_HOME
from paths import CHAR_VOCAB_HOME
from paths import MODELS_HOME
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--save-to", dest="save_to", type=str, default=None)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([128]))
parser.add_argument("--char_embedding_size", type=int, default=random.choice([100]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([1]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0]))
parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0]))
parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))


import math

args=parser.parse_args()

if args.save_to is not None and "MYID" in args.save_to:
   args.save_to = args.save_to.replace("MYID", str(args.myID))

print(args)



import corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

with open(CHAR_VOCAB_HOME+"/char-vocab-wiki-"+args.language, "r") as inFile:
  itos = inFile.read().strip().split("\n")
#itos = sorted(itos)
print(itos)
stoi = dict([(itos[i],i) for i in range(len(itos))])




import random


import torch

print(torch.__version__)

from weight_drop import WeightDrop


rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()
rnn_decoder = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()


rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = rnn #WeightDrop(rnn, [(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

char_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.char_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

hiddenToMean = torch.nn.Linear(args.hidden_dim, 2*args.hidden_dim).cuda()
hiddenToLogSD = torch.nn.Linear(args.hidden_dim, 2*args.hidden_dim).cuda()

modules = [rnn, output, char_embeddings, rnn_decoder, hiddenToMean, hiddenToLogSD]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

named_modules = {"rnn" : rnn, "output" : output, "char_embeddings" : char_embeddings, "optim" : optim}

if args.load_from is not None:
  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
  for name, module in named_modules.items():
      module.load_state_dict(checkpoint[name])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout


def prepareDatasetChunks(data, train=True):
      words = []
      count = 0
      print("Prepare chunks")
      numerified = []
      for chunk in data:
       print(len(chunk))
       for word in chunk:
         words.append(word)
         if len(words) == args.batchSize:
            yield words
            words = []

hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

def encodeWord(word):
      numeric = [[]]
      for char in word:
           numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      return numeric

standardNormal = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(2*args.hidden_dim)] for _ in range(args.batchSize)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(2*args.hidden_dim)] for _ in range(args.batchSize)]).cuda())
standardNormalPerStep = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(2*args.hidden_dim)] for _ in range(args.batchSize)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(2*args.hidden_dim)] for _ in range(args.batchSize)]).cuda())
standardNormalPerStepTwo = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(2*args.hidden_dim)] for _ in range(2)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(2*args.hidden_dim)] for _ in range(2)]).cuda())
standardNormalPerStepOne = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(2*args.hidden_dim)] for _ in range(1)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(2*args.hidden_dim)] for _ in range(1)]).cuda())


def forward(words, printHere=False, train=False):
    numeric = [encodeWord(word)[0] for word in words]
    maxLength = max([len(x) for x in numeric])
    charCount = sum([len(x)+1 for x in numeric])
    numericIn = [None for _ in numeric]
    numericOut = [None for _ in numeric]
    for i in range(len(numeric)):
       numericIn[i] = ([0]*(maxLength-len(numeric[i]))) + numeric[i]
       numericOut[i] = numeric[i] + [1] + ([0]*(maxLength-len(numeric[i])))

    input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numericIn]).transpose(0,1).cuda(), requires_grad=False)
    input_tensor_decoder = Variable(torch.LongTensor([[0]+x for x in numericOut]).transpose(0,1).cuda(), requires_grad=False)
    embedded_forward = char_embeddings(input_tensor_forward)
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)


    mean = hiddenToMean(hidden_forward[0])
    logSD = hiddenToLogSD(hidden_forward[0])

    memoryDistribution = torch.distributions.Normal(loc=mean, scale=torch.exp(logSD))# replaced exponential by tanh to fight NaN problems

    encodedEpsilon = standardNormalPerStep.sample()
    sampled = mean + torch.exp(logSD) * encodedEpsilon # replaced exponential by tanh to fight NaN problems

    logProbConditional = memoryDistribution.log_prob(sampled).sum(dim=2).mean()  # TODO not clear whether back-prob through sampled?


    plainPriorLogProb = standardNormal.log_prob(sampled).sum(dim=2).mean() #- (0.5 * torch.sum(sampled * sampled, dim=1))
    hidden_forward = (sampled[:,:,:args.hidden_dim].contiguous(), sampled[:,:,args.hidden_dim:].contiguous())


    if printHere:
       print("LIKELIHOOD RATIO", (plainPriorLogProb - logProbConditional))

#    hidden_forward = (sample

    out_decoder, _ = rnn_decoder(char_embeddings(input_tensor_decoder[:-1]), hidden_forward)
    softmaxDecoder = logsoftmax(output(out_decoder))
    target_tensor = input_tensor_decoder[1:]
    loss = train_loss(softmaxDecoder.view(-1, len(itos)+3), target_tensor.view(-1))
    loss = loss - 0.0001 * (plainPriorLogProb - logProbConditional)
    if printHere:
       print("Train loss here", loss)
    return loss, charCount





def backward(loss, printHere):
      optim.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
      optim.step()










from torch.autograd import Variable


def encodeWord(word):
      numeric = [[]]
      for char in word:
           numeric[-1].append((stoi[char]+3 if char in stoi else 2) if True else 2+random.randint(0, len(itos)))
      return numeric


def encodeSequenceBatchForward(numeric):
      input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numeric]).transpose(0,1).cuda(), requires_grad=False)

#      target_tensor_forward = Variable(torch.LongTensor(numeric).transpose(0,1)[2:].cuda(), requires_grad=False).view(args.sequence_length+1, len(numeric), 1, 1)
      embedded_forward = char_embeddings(input_tensor_forward)
      out_forward, hidden_forward = rnn_forward_drop(embedded_forward, None)
#      out_forward = out_forward.view(args.sequence_length+1, len(numeric), -1)
 #     logits_forward = output(out_forward) 
  #    log_probs_forward = logsoftmax(logits_forward)
      return (out_forward[-1], hidden_forward)




import numpy as np


def decode(words, printHere=False, train=False):
    numeric = [encodeWord(word)[0] for word in words]
    maxLength = max([len(x) for x in numeric])
    charCount = sum([len(x)+1 for x in numeric])
    numericIn = [None for _ in numeric]
    numericOut = [None for _ in numeric]
    for i in range(len(numeric)):
       numericIn[i] = ([0]*(maxLength-len(numeric[i]))) + numeric[i]
       numericOut[i] = numeric[i] + [1] + ([0]*(maxLength-len(numeric[i])))

    input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numericIn]).transpose(0,1).cuda(), requires_grad=False)
    input_tensor_decoder = Variable(torch.LongTensor([[0]+x for x in numericOut]).transpose(0,1).cuda(), requires_grad=False)
    embedded_forward = char_embeddings(input_tensor_forward)
    #print("numeric", embedded_forward.size())
    out_forward, hidden_forward = rnn_drop(embedded_forward, None)

    mean = hiddenToMean(hidden_forward[0])
    logSD = hiddenToLogSD(hidden_forward[0])
    encodedEpsilon = standardNormalPerStepOne.sample()
    sampled = mean + torch.exp(logSD) * encodedEpsilon # replaced exponential by tanh to fight NaN problems
    hidden_forward = (sampled[:,:,:args.hidden_dim].contiguous(), sampled[:,:,args.hidden_dim:].contiguous())






    output_string = ""
    lastOutput = torch.zeros(len(numericIn)).long().cuda()
    for _ in range(10):
      embedded = char_embeddings(lastOutput).unsqueeze(0)
      out_decoder, hidden_forward = rnn_decoder(embedded, hidden_forward)
      softmaxDecoder = logsoftmax(output(out_decoder))

      prediction = softmaxDecoder.view(3+len(itos)).cpu().detach().numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
#      predicted = np.argmax(prediction).items()
      predicted = np.random.choice(3+len(itos), p=np.exp(prediction))

      output_string += itos[predicted-3]
      lastOutput = torch.Tensor([predicted]).long().cuda()


    return output_string 








def decode_mix(words, printHere=False, train=False):
    numeric = [encodeWord(word)[0] for word in words]
    maxLength = max([len(x) for x in numeric])
    charCount = sum([len(x)+1 for x in numeric])
    numericIn = [None for _ in numeric]
    numericOut = [None for _ in numeric]
    for i in range(len(numeric)):
       numericIn[i] = ([0]*(maxLength-len(numeric[i]))) + numeric[i]
       numericOut[i] = numeric[i] + [1] + ([0]*(maxLength-len(numeric[i])))

    input_tensor_forward = Variable(torch.LongTensor([[0]+x for x in numericIn]).transpose(0,1).cuda(), requires_grad=False)
    input_tensor_decoder = Variable(torch.LongTensor([[0]+x for x in numericOut]).transpose(0,1).cuda(), requires_grad=False)
    embedded_forward = char_embeddings(input_tensor_forward)
    #print("numeric", embedded_forward.size())
    out_forward, hidden_forward_fromEnc = rnn_drop(embedded_forward, None)

    mean = hiddenToMean(hidden_forward_fromEnc[0])
    logSD = hiddenToLogSD(hidden_forward_fromEnc[0])
    encodedEpsilon = standardNormalPerStepTwo.sample()
    sampled = mean + torch.exp(logSD) * encodedEpsilon # replaced exponential by tanh to fight NaN problems
    hidden_forward_fromEnc = (sampled[:,:,:args.hidden_dim].contiguous(), sampled[:,:,args.hidden_dim:].contiguous())






    for _ in range(10):
       hidden_forward = list(hidden_forward_fromEnc)
       hidden_forward[0] = hidden_forward[0].mean(dim=1, keepdim=True)
       hidden_forward[1] = hidden_forward[1].mean(dim=1, keepdim=True)

       output_string = ""
       lastOutput = torch.zeros(1).long().cuda()
       for _ in range(10):
         embedded = char_embeddings(lastOutput).unsqueeze(0)
         out_decoder, hidden_forward = rnn_decoder(embedded, hidden_forward)
         softmaxDecoder = logsoftmax(output(out_decoder))
   
         prediction = softmaxDecoder.view(3+len(itos)).cpu().detach().numpy() #.view(1,1,-1))).view(3+len(itos)).data.cpu().numpy()
   #      predicted = np.argmax(prediction).items()
         predicted = np.random.choice(3+len(itos), p=np.exp(prediction))
   
         output_string += itos[predicted-3]
         lastOutput = torch.Tensor([predicted]).long().cuda()

       print(output_string)
    








import time

devLosses = []
try:
 for epoch in range(10000):
   print(epoch)
   training_data = corpusIteratorWikiWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   while True:
      counter += 1
      try:
         numeric = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      trainChars += charCounts 
      if printHere:
          print((epoch,counter))
          print("Dev losses")
          print(devLosses)
          print("Chars per sec "+str(trainChars/(time.time()-startTime)))
          print(learning_rate)
          print(args)

          
          #print(keepGenerating(encodeSequenceBatchForward(encodeWord(" ich mach"))))
          print("=========================")
          print(decode(["person"]))
          print(decode(["people"]))
          decode_mix(["person", "people"])

          print("=========================")
         
          


      if counter % 20000 == 0 and epoch == 0:
        if args.save_to is not None:
           torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")


   rnn_drop.train(False)


   dev_data = corpusIteratorWikiWords.dev(args.language)
   print("Got data")
   dev_chars = prepareDatasetChunks(dev_data, train=False)


     
   dev_loss = 0
   dev_char_count = 0
   counter = 0
   hidden, beginning = None, None
   while True:
       counter += 1
       try:
          numeric = next(dev_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)
   with open(LOG_HOME+"/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as outFile:
      print(" ".join([str(x) for x in devLosses]), file=outFile)
      print(" ".join(sys.argv), file=outFile)
      print(str(args), file=outFile)
   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break
   if args.save_to is not None:
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")

   learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
   optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9
except KeyboardInterrupt:
   if args.save_to is not None:
      print("Saving")
      torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")



