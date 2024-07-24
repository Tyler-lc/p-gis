import sys
import os

sys.path.append("c:\\VSCode_python\\p_gis\\p-gis")

from tests.test_lib import defineArguments, processInput
from tests.test_functions.create_network_test import createNetworkTest


availableTests = {"create_network": createNetworkTest}


def init():
    args = defineArguments(availableTests)

    processInput(args, availableTests)


if __name__ == "__main__":
    init()
