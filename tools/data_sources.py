from tools.PubmedDataloader import PubmedDataloader


# Dictionary that matches all data_sources to a key
# To be used in the DataLoader class
data_sources = {
    "pub_med": PubmedDataloader("http://localhost:5000")
}