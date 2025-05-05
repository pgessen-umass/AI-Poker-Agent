import pprint
from pypokerengine.api.game import setup_config, start_poker
from q.game_encoder import PokerGameEncoder
from q.model import DeepQModelWrapper, QModel
from q.training_player import TrainingPlayer
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=10)

model1 = QModel()
wrapper1 = DeepQModelWrapper(model1, 0.9, 0.2, "dumps", 100, 1000)
model2 = QModel()
wrapper2 = DeepQModelWrapper(model1, 0.9, 0.2, "dumps", 10)
p1 = TrainingPlayer(wrapper1, verbose=True)
p2 = TrainingPlayer(wrapper2, verbose=False)

# p2 = DeepQPlayer(counter, PlayerType.RESETTER)

# config.register_player(name="f1", algorithm=RandomPlayer())
# config.register_player(name="FT2", algorithm=RaisedPlayer())
config.register_player(name="DeepQ Player", algorithm=p1)
config.register_player(name="Random Player", algorithm=RandomPlayer())

game_result = start_poker(config, verbose=0)

pprint.pprint(game_result)