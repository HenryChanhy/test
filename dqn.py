from sys import argv
from os.path import join, dirname
import codecs
from numpy import zeros, random, maximum, array, argmax, nan
from keras.optimizers import SGD

from environment import MarketEnv
from market_model_builder import MarketModelBuilder


BASE_DIR = dirname(__file__)

from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = '/content/stock_market_reinforcement_learning/cert.json'

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#print(credentials)

from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
drive_service = build('drive', 'v3', credentials=credentials)
def savemodel(name):
  file_metadata = {
  'name': name+'.json',
  'mimeType': 'text/plain'
  }
  #obj.to_csv("/tmp/"+name+".csv",encoding = 'utf-8')
  media = MediaFileUpload("/content/stock_market_reinforcement_learning/models/"+name+".json",
                        mimetype='text/plain',
                        resumable=True)
  created = drive_service.files().update(body=file_metadata,
                                       media_body=media,
                                       fileId='12Fp4_KYdHc_RxYgz3Gz4vGQMNYM4FC7U').execute()
  print('Filename:{} File ID: {}'.format(name,created.get('id')))
  file_metadata = {
  'name': name+'.h5',
  'mimeType': 'text/plain'
  }
  media = MediaFileUpload("/content/stock_market_reinforcement_learning/models/"+name+".h5",
                        mimetype='text/plain',
                        resumable=True)
  created1 = drive_service.files().update(body=file_metadata,
                                       media_body=media,
                                       fileId='1Z9UC8wGmwu4PL6bRGQiwhuzEJlrU3cUd').execute()
  print('Filename:{} File ID: {}'.format(name,created1.get('id')))
  return created.get('id'),created1.get('id')

def loaddate(id,filename):
  request = drive_service.files().get_media(fileId=id)
  #downloaded = io.BytesIO()
  fh = io.FileIO("/content/stock_market_reinforcement_learning/models/"+filename, 'wb')
  downloader = MediaIoBaseDownload(fh, request)
  done = False
  while done is False:
    # _ is a placeholder for a progress object that we ignore.
    # (Our file is small, so we skip reporting progress.)
    _, done = downloader.next_chunk()

  fh.seek(0)

  #print('Downloaded file contents are: {}'.format(downloaded.read()))
  return fh


def restoremodle():
  results = drive_service.files().list(
      pageSize=10, fields="nextPageToken, files(id, name)").execute()
  items = results.get('files', [])
  if not items:
      print('No files found.')
  else:
      print('Files:')
      for item in items:
        loaddate(item['id'],item['name'])
        print('restoring model {0} ({1})'.format(item['name'], item['id']))

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        inputs = []

        dim = len(self.memory[0][0][0])
        for i in range(dim):
            inputs.append([])

        targets = zeros((min(len_memory, batch_size), num_actions))
        for i, idx in enumerate(random.randint(0, len_memory, size=min(len_memory, batch_size))):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            for j in range(dim):
                inputs[j].append(state_t[j][0])

            #inputs.append(state_t)
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa

        #inputs = array(inputs)
        inputs = [array(inputs[i]) for i in range(dim)]

        return inputs, targets


if __name__ == "__main__":
    portfolio_list = argv[1]
    model_filename = argv[2] if len(argv) > 2 else None

    instruments = {}
    f = codecs.open(portfolio_list, "r", "utf-8")

    for line in f:
        if line.strip() != "" and line.strip() != 'code,name':
            tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
            instruments[tokens[0]] = tokens[1]

    f.close()

    env = MarketEnv(target_symbols=list(instruments.keys()), input_symbols=[],
        start_date="1980-01-01",
        end_date="2018-06-29",
        sudden_death=-1.0)

    # parameters
    epsilon = 0.5  # exploration
    min_epsilon = 0.1
    epoch = 100000
    max_memory = 5000
    batch_size = 128
    discount = 0.8

    model = MarketModelBuilder(join(BASE_DIR, "models", "model.h5") if model_filename == None else join(BASE_DIR, "models", model_filename + ".h5")).getModel()
    sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss='mse', optimizer='rmsprop')

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory = max_memory, discount = discount)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.reset()
        cumReward = 0

        while not game_over:
            input_tm1 = input_t
            isRandom = False

            # get next action
            if random.rand() <= epsilon:
                action = random.randint(0, env.action_space.n, size=1)[0]

                isRandom = True
            else:
                q = model.predict(input_tm1)
                action = argmax(q[0])

                #print "  ".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())])
                if nan in q:
                    print("I found NaN!")
                    exit()

            # apply action, get rewards and new state
            input_t, reward, game_over, info = env.step(action)
            cumReward += reward

            if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
                color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
                if isRandom:
                    color = bcolors.WARNING if env.actions[action] == "LONG" else bcolors.OKGREEN
                print(("%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], color + env.actions[action] + bcolors.ENDC, cumReward, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())]) if isRandom == False else "")))

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

        if cumReward > 0 and game_over:
            win_cnt += 1

        print(("Epoch {:03d}/{} | Loss {:.4f} | Win count {} | Epsilon {:.4f}".format(e, epoch, loss, win_cnt, epsilon)))
        # Save trained model weights and architecture, this will be used by the visualization code
        model_json = model.to_json()
        with open(join(BASE_DIR, "models", model_filename + ".json"), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(join(BASE_DIR, "models", "model.h5") if model_filename == None else join(BASE_DIR, "models", model_filename + ".h5"), overwrite=True)
        epsilon = maximum(min_epsilon, epsilon * 0.99)
