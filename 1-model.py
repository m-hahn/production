import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", dest="batchSize", type=int, default=32)

args=parser.parse_args()
print(args)



# goal representation
# per-phoneme producer


# push sentence into goal
# 

dist = torch.distributions.bernoulli.Bernoulli(torch.tensor([[0.03 for _ in range(500)] for _ in range(args.batchSize)]))

encoder = torch.nn.Linear(500, 100)

embeddings = torch.nn.Embedding(100, 100)
speaker = torch.nn.LSTM(input_size=200, hidden_size=50)
listener = torch.nn.LSTM(input_size=200, hidden_size=50)
decoder = torch.nn.Linear(50, 100)

def randomChoice(ps):
               uniformWeights = torch.rand(ps.size()[0])
               chosen = torch.zeros(ps.size()[0]).long()
               massSoFar = torch.zeros(ps.size()[0])
               covered = torch.zeros(uniformWeights.size()[0]).byte()
               for j in range(ps.size()[1]):
                  massSoFar += ps[:,j]
                  chosen[(massSoFar >= uniformWeights) * (1- covered)] = j
                  covered[massSoFar >= uniformWeights] = 1
               return chosen



for i in range(1):
   inputs = dist.sample()
   print(inputs.size())
   hidden = encoder(inputs)
   print(hidden.size())   

   encoded = hidden

   lstm_hidden = None
   message = [torch.zeros(args.batchSize).long()]
   message_encoded = [embeddings(message[-1])]
   for j in range(5):
      inp = torch.cat([message_encoded[-1], encoded], dim=1).unsqueeze(0)
      output, lstm_hidden = speaker(inp, lstm_hidden)
      output = output.squeeze(0)
      print(output.size())
      print(decoder(output).size())
      logsoftmax = torch.nn.functional.log_softmax(decoder(output), dim=1)
      nextWord = randomChoice(torch.exp(logsoftmax).squeeze(0))
      message.append(nextWord)
      message_encoded.append(embeddings(message[-1]))
   print(message)
   print([x.size() for x in message_encoded])


   message_encoded = torch.stack(message_encoded)
   print(message_encoded.size(), hidden.unsqueeze(0).size())
   inp = torch.cat([message_encoded, hidden.unsqueeze(0).expand(message_encoded.size()[0], -1, -1)], dim=2)
   print(inp.size())
   _, listener_hidden = listener(inp, None)
   print(listener_hidden[0].size())


