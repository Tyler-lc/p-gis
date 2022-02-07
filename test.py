from .tests.test_lib import defineArguments, processInput
from .tests.functions.create_network_test import createNetworkTest


availableTests = {"create_network": createNetworkTest}


def init():
    args = defineArguments(availableTests)

    processInput(args, availableTests)
