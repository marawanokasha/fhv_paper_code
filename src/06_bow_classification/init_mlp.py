import os
import sys

sys.path.append(os.path.abspath('..'))
from utils.metrics import get_metrics, get_binary_0_5
from utils.classification import get_label_data
from utils.file import ensure_disk_location_exists

from classification_util import *

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder


NN_SEED = 1234
np.random.seed(NN_SEED)

MAX_TERMS = 10000

GLOBAL_VARS = namedtuple('GLOBAL_VARS', ['MODEL_NAME', 'NN_MODEL_NAME'])
NN_PARAMETER_SEARCH_PREFIX = "nn_bow_{}_batch_{}_nn_parameter_searches.pkl"


root_location = "../../data/"
exports_location = root_location + "exported_data/"
nn_parameter_search_location = os.path.join(root_location, "nn_bow_parameter_search")

classifications_index_file = os.path.join(exports_location, "classifications_index.pkl")
doc_classification_map_file = os.path.join(exports_location, "doc_classification_map.pkl")
sections_file = os.path.join(exports_location, "sections.pkl")
valid_classes_file = os.path.join(exports_location, "valid_classes.pkl")
valid_subclasses_file = os.path.join(exports_location, "valid_subclasses.pkl")
classifications_file = os.path.join(exports_location, "classifications.pkl")
doc_lengths_map_file = os.path.join(exports_location, "doc_lengths_map.pkl")
training_docs_list_file = os.path.join(exports_location, "training_docs_list.pkl")
validation_docs_list_file = os.path.join(exports_location, "validation_docs_list.pkl")
test_docs_list_file = os.path.join(exports_location, "test_docs_list.pkl")


## Load utility data

doc_classification_map = pickle.load(open(doc_classification_map_file))
sections = pickle.load(open(sections_file))
valid_classes = pickle.load(open(valid_classes_file))
valid_subclasses = pickle.load(open(valid_subclasses_file))
training_docs_list = pickle.load(open(training_docs_list_file))
validation_docs_list = pickle.load(open(validation_docs_list_file))
test_docs_list = pickle.load(open(test_docs_list_file))


classification_types = {
    "sections": sections,
    "classes": valid_classes,
    "subclasses": valid_subclasses
}
possible_data_types = ["tf", "sublinear_tf", "tf_idf", "sublinear_tf_idf","bm25"]
activations = ['relu','sigmoid', 'tanh']

# specify on command line the SVM parameters and the bow representation to use

parser = argparse.ArgumentParser(description='Run MLP on BOW data')
parser.add_argument("-c", "--classificationsType", choices=classification_types.keys(), required=True)
parser.add_argument("-d", "--dataType", choices=possible_data_types, required=True)
parser.add_argument("-t", "--doTest", action="store_true", help="Whether to do testing or parameter searching")
parser.add_argument("--testInputDropout", required=False, help="Input Dropout")
parser.add_argument("--test1stActivation", required=False, help="1st layer activation function", choices=activations)
parser.add_argument("--test1stSize", required=False, help="1st layer size", type=int)
parser.add_argument("--test1stDropout", action="store_true", help="1st layer dropout")
parser.add_argument("--test2ndActivation", required=False, help="2nd layer activation function", choices=activations)
parser.add_argument("--test2ndSize", required=False, help="2nd layer size", type=int)
parser.add_argument("--test2ndDropout", action="store_true", help="2nd layer dropout")
args = parser.parse_args()

print args

classifications_type = args.classificationsType
classifications = classification_types[classifications_type]
data_type = args.dataType
DO_TEST = args.doTest


data_training_location = os.path.join(exports_location, "{}_training_sparse_data.pkl".format(data_type))
data_training_docids_location = os.path.join(exports_location, "{}_training_sparse_docids.pkl".format(data_type))
data_validation_location = os.path.join(exports_location, "{}_validation_sparse_data.pkl".format(data_type))
data_validation_docids_location = os.path.join(exports_location, "{}_validation_sparse_docids.pkl".format(data_type))
data_test_location = os.path.join(exports_location, "{}_test_sparse_data.pkl".format(data_type))
data_test_docids_location = os.path.join(exports_location, "{}_test_sparse_docids.pkl".format(data_type))


NN_OUTPUT_NEURONS = len(classifications)
EARLY_STOPPER_MIN_DELTA = 0.00001
EARLY_STOPPER_PATIENCE = 15
NN_MAX_EPOCHS = 200
NN_RANDOM_SEARCH_BUDGET = 20
NN_PARAM_SAMPLE_SEED = 1234

NN_BATCH_SIZE = 2048

MODEL_VERBOSITY = 1

load_existing_results = True
save_results = True


if DO_TEST == False:

    first_hidden_layer_sizes = [100,200,500,1000]
    second_hidden_layer_sizes = [None,500,1000,2000]
    first_hidden_layer_activations = ['relu','sigmoid', 'tanh']
    second_hidden_layer_activations = ['relu','sigmoid', 'tanh']
    input_dropout_options = [False]
    hidden_dropout_options = [True]
    second_hidden_dropout_options = [False]

    info("=============== {} Being Evaluated with a parameter search ================".format(data_type))

    GLOBAL_VARS.MODEL_NAME = data_type + "/size_{}".format(MAX_TERMS)

    # Get the training data
    info('Getting Training Data')
    X = pickle.load(open(data_training_location, "r"))
    training_data_docids = pickle.load(open(data_training_docids_location, "r"))
    y = get_label_data(classifications, training_data_docids, doc_classification_map)

    print X.shape
    print y.shape

    # Get the validation data
    info('Getting Validation Data')
    Xv = pickle.load(open(data_validation_location,'r'))
    validation_data_docids = pickle.load(open(data_validation_docids_location, "r"))
    yv = get_label_data(classifications, validation_data_docids, doc_classification_map)

    NN_INPUT_NEURONS = X.shape[1]


    param_sampler = ParameterSampler({
        'first_hidden_layer_size':first_hidden_layer_sizes,
        'first_hidden_layer_activation':first_hidden_layer_activations,
        'second_hidden_layer_size':second_hidden_layer_sizes,
        'second_hidden_layer_activation':second_hidden_layer_activations,
        'input_dropout':input_dropout_options,
        'hidden_dropout':hidden_dropout_options,
        'second_hidden_dropout':second_hidden_dropout_options
    }, n_iter=NN_RANDOM_SEARCH_BUDGET, random_state=NN_PARAM_SAMPLE_SEED)

    param_results_dict = {}

    param_results_path = os.path.join(os.path.join(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME,
                                           NN_PARAMETER_SEARCH_PREFIX.format(classifications_type, NN_BATCH_SIZE)))

    if load_existing_results:
        param_results_path = os.path.join(os.path.join(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME,
                                           NN_PARAMETER_SEARCH_PREFIX.format(classifications_type, NN_BATCH_SIZE)))
        if os.path.exists(param_results_path):
            info('Loading Previous results in {}'.format(param_results_path))
            param_results_dict = pickle.load(open(param_results_path))
        else:
            info('No Previous results exist in {}'.format(param_results_path))

    ensure_disk_location_exists(os.path.join(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME))

    for parameters in param_sampler:
        start_time = time.time()
        first_hidden_layer_size = parameters['first_hidden_layer_size']
        first_hidden_layer_activation = parameters['first_hidden_layer_activation']
        second_hidden_layer_size = parameters['second_hidden_layer_size']
        second_hidden_layer_activation = parameters['second_hidden_layer_activation']
        input_dropout_do = parameters['input_dropout']
        hidden_dropout_do = parameters['hidden_dropout']
        second_hidden_dropout_do = parameters['second_hidden_dropout']

        GLOBAL_VARS.NN_MODEL_NAME = 'nn_1st-size_{}_1st-act_{}_2nd-size_{}_2nd-act_{}_in-drop_{}_hid-drop_{}'.format(
            first_hidden_layer_size, first_hidden_layer_activation, second_hidden_layer_size,
            second_hidden_layer_activation, input_dropout_do, hidden_dropout_do
        )
        if second_hidden_dropout_do:
            GLOBAL_VARS.NN_MODEL_NAME = GLOBAL_VARS.NN_MODEL_NAME + '_2nd-hid-drop_{}'.format(str(second_hidden_dropout_do))

        if GLOBAL_VARS.NN_MODEL_NAME in param_results_dict.keys():
            print "skipping: {}".format(GLOBAL_VARS.NN_MODEL_NAME)
            continue

        info('***************************************************************************************')
        info(GLOBAL_VARS.NN_MODEL_NAME)

        model = create_keras_nn_model(NN_INPUT_NEURONS, NN_OUTPUT_NEURONS,
                                      first_hidden_layer_size, first_hidden_layer_activation,
                                      second_hidden_layer_size, second_hidden_layer_activation,
                                      input_dropout_do, hidden_dropout_do, second_hidden_dropout_do)
        model.summary()

        early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=EARLY_STOPPER_MIN_DELTA, \
                                                      patience=EARLY_STOPPER_PATIENCE, verbose=1, mode='auto')
        metrics_callback = MetricsCallbackWithGenerator(classifications_type, NN_BATCH_SIZE, Xv, yv)

        history = model.fit_generator(generator=nn_batch_generator(X, y, NN_BATCH_SIZE), \
                                            samples_per_epoch=X.shape[0],\
                                            validation_data=nn_batch_generator(Xv, yv, NN_BATCH_SIZE),\
                                            nb_val_samples=Xv.shape[0],\
                                            verbose=MODEL_VERBOSITY,\
                                            nb_epoch=NN_MAX_EPOCHS, callbacks=[early_stopper, metrics_callback])

        # using the recorded weights of the best recorded validation loss
        last_model_weights = model.get_weights()
        info('Evaluating on Validation Data using saved best weights')
        model.set_weights(metrics_callback.best_weights)
        yvp = model.predict_generator(generator=nn_batch_generator(Xv, yv, NN_BATCH_SIZE),
                                                   val_samples=Xv.shape[0])
        yvp_binary = get_binary_0_5(yvp)
        info('Generating Validation Metrics')
        validation_metrics = get_metrics(yv, yvp, yvp_binary)
        print "****** Validation Metrics: Cov Err: {:.3f} | Top 3: {:.3f} | Top 5: {:.3f} | F1 Micro: {:.3f} | F1 Macro: {:.3f}".format(
            validation_metrics['coverage_error'], validation_metrics['top_3'], validation_metrics['top_5'],
            validation_metrics['f1_micro'], validation_metrics['f1_macro'])
        best_validation_metrics = validation_metrics
        time.sleep(0.2)

        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME] = dict()
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_validation_metrics'] = best_validation_metrics
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['epochs'] = len(history.history['val_loss'])
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_weights'] = metrics_callback.best_weights
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_val_loss'] = metrics_callback.best_val_loss
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['training_loss'] = metrics_callback.losses
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['validation_loss'] = metrics_callback.val_losses

        duration = time.time() - start_time
        param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['duration'] =  duration

        del history, last_model_weights, metrics_callback

    if save_results:
        pickle.dump(param_results_dict, open(os.path.join(os.path.join(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME,
                                                                       NN_PARAMETER_SEARCH_PREFIX.format(classifications_type, NN_BATCH_SIZE))), 'w'))

else:
    info('Doing Testing')
    TEST_METRICS_FILENAME = '{}_batch_{}_test_metrics.pkl'.format(classifications_type, NN_BATCH_SIZE)
    GLOBAL_VARS.MODEL_NAME = data_type + "/size_{}".format(MAX_TERMS)
    param_results_dict = pickle.load(open(os.path.join(os.path.join(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME,
                                       NN_PARAMETER_SEARCH_PREFIX.format(classifications_type, NN_BATCH_SIZE)))))
    # Get the test data
    info('Getting Test Data')
    Xt = pickle.load(open(data_test_location, "r"))
    test_data_docids = pickle.load(open(data_test_docids_location, "r"))
    yt = get_label_data(classifications, test_data_docids, doc_classification_map)

    NN_INPUT_NEURONS = Xt.shape[1]


    first_hidden_layer_size = args.test1stSize
    first_hidden_layer_activation = args.test1stActivation
    second_hidden_layer_size = args.test1stActivation
    second_hidden_layer_activation = args.test1stActivation
    input_dropout_do = args.testInputDropout
    hidden_dropout_do = args.test1stDropout
    second_hidden_dropout_do = args.test2ndDropout


    GLOBAL_VARS.NN_MODEL_NAME = 'nn_1st-size_{}_1st-act_{}_2nd-size_{}_2nd-act_{}_in-drop_{}_hid-drop_{}'.format(
        first_hidden_layer_size, first_hidden_layer_activation, second_hidden_layer_size,
        second_hidden_layer_activation, input_dropout_do, hidden_dropout_do
    )
    if second_hidden_dropout_do:
        GLOBAL_VARS.NN_MODEL_NAME = GLOBAL_VARS.NN_MODEL_NAME + '_2nd-hid-drop_{}'.format(str(second_hidden_dropout_do))
    if GLOBAL_VARS.NN_MODEL_NAME not in param_results_dict.keys():
        print "Can't find model: {}".format(GLOBAL_VARS.NN_MODEL_NAME)
        raise Exception()

    info('***************************************************************************************')
    info(GLOBAL_VARS.NN_MODEL_NAME)

    model = create_keras_nn_model(NN_INPUT_NEURONS, NN_OUTPUT_NEURONS,
                                  first_hidden_layer_size, first_hidden_layer_activation,
                                  second_hidden_layer_size, second_hidden_layer_activation,
                                  input_dropout_do, hidden_dropout_do, second_hidden_dropout_do)
    model.summary()

    # get model best weights
    weights = param_results_dict[GLOBAL_VARS.NN_MODEL_NAME]['best_weights']
    model.set_weights(weights)

    time.sleep(0.2)
    info('Evaluating on Test Data using best weights')
    ytp = model.predict_generator(generator=nn_batch_generator(Xt, yt, NN_BATCH_SIZE), val_samples=Xt.shape[0])
    ytp_binary = get_binary_0_5(ytp)
    info('Generating Test Metrics')
    test_metrics = get_metrics(yt, ytp, ytp_binary)
    print "** Test Metrics: Cov Err: {:.3f}, Avg Labels: {:.3f}, \n\t\t Top 1: {:.3f}, Top 3: {:.3f}, Top 5: {:.3f}, \n\t\t F1 Micro: {:.3f}, F1 Macro: {:.3f}, Total Pos: {:,d}".format(
        test_metrics['coverage_error'], test_metrics['average_num_of_labels'],
        test_metrics['top_1'], test_metrics['top_3'], test_metrics['top_5'],
        test_metrics['f1_micro'], test_metrics['f1_macro'], test_metrics['total_positive'])

    ensure_disk_location_exists(os.path.join(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME))

    pickle.dump(test_metrics, open(os.path.join(nn_parameter_search_location, GLOBAL_VARS.MODEL_NAME,
                                                TEST_METRICS_FILENAME), 'w'))
