from tools.PubmedDataLoader import PubmedDataLoader


# Dictionary that matches all data_sources to a key
# To be used in the DataLoader class
data_sources = {
    "pub_med": PubmedDataLoader("http://localhost:5112")
}