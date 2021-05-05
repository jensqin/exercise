from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import awswrangler as wr

from settings import S3_FOLDER


def encoder_from_s3(name):
    """get encoder from s3"""
    if name == "team":
        team_map = wr.s3.read_csv(S3_FOLDER + "encoders/team_map_all.csv")
        encoder = OrdinalEncoder()
        encoder.fit(team_map[["teamids"]])
    elif name == "player":
        player_map = wr.s3.read_csv(S3_FOLDER + "encoders/player_map_all.csv")
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(player_map[["playerids"]])
    elif name == "event":
        event_map = wr.s3.read_csv(S3_FOLDER + "encoders/event_map_all.csv")
        encoder = OneHotEncoder()
        encoder.fit(event_map[["events"]])
    else:
        raise ValueError(f"{name} encoder does not exist.")
    return encoder
