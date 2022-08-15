from io import BytesIO
import pickle
import requests
import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
import pandas as pd

PICKLE_FILE_NAME = 'model/rf_model.pkl'
DATA_FILE_NAME = 'census_data/data.csv'

#workflow:
# 1. check if session state contains a random forest model.
# 2. if not, try to load from a pickle file.
# 3. if pickle file is not available, load from github.

#allocates random forest model to session state.
def main(st):
    try:
        st.session_state['model_rf']
        logging.info('session state contains random forest model.')
    except:

        try:
            model_rf = pickle.load(open(PICKLE_FILE_NAME, 'rb'))
            st.session_state['model_rf'] = model_rf
            logging.info('loaded model from directory and saved to session.')

        except:
            mLink = 'https://raw.githubusercontent.com/kayfilipp/HigherME/main/census_data/model_export/finalized_rf_model.pkl?raw=true'
            mfile = BytesIO(requests.get(mLink).content)
            model_rf = pickle.load(mfile)
            st.session_state['model_rf'] = model_rf

            logging.info('saving model to local directory from github.')
            pickle.dump(st.session_state['model_rf'], open(PICKLE_FILE_NAME, 'wb'))

            del mLink
            del mfile
        finally:

            # save RF model to session state to avoid reloading every time we open.
            st.session_state['model_rf'] = model_rf
            del model_rf

#loads data.csv - used for selection boxes and dummification, etc.
#workflow
# 1. check session state
# 2. check local dir
# 3. check github, save to local

#stores data.csv in session state
def main_data(st):
    try:
        st.session_state['data']
        logging.info('session state contains dataframe.')
    except:
        logging.info('session does not contain data.csv. pulling...')
        try:
            st.session_state['data'] = pd.read_csv('census_data/data.csv')
            logging.info('loaded data.csv from directory and saved to session')
        except:
            data = pd.read_csv("https://raw.githubusercontent.com/kayfilipp/HigherME/main/census_data/streamlit/data.csv")

            logging.info('writing data.csv locally from github.')
            data.to_csv(DATA_FILE_NAME,index=None)
            logging.info('saving data to session state from github.')
            st.session_state['data'] = data
            del data


#used to convert radio inputs for sex, is_stem, birthplace, racial group
def converter_func(radio):
    if radio=='yes' or radio=='Female':
        return 1
    return 0
