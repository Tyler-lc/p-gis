import argparse
import os
import sys
from colorama import init
from colorama.ansi import Back, Fore, Style


def initTests():
    init()


def blockPrint():
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__


def printToDocument(args, file):
    sys.stdout = open(os.path.join(args.dir, file + ".log"), "w")


def processOutputBefore(args, test):
    if not args.show_outputs:
        blockPrint()
    if args.show_outputs and args.dir:
        printToDocument(args, test)


def processOutputAfter():
    enablePrint()


def printError(message):
    print(Fore.RED + Back.WHITE + " ERROR " + Style.RESET_ALL + " " + message)


def printSuccess(message):
    print(Fore.GREEN + Back.WHITE + " SUCCESS " + Style.RESET_ALL + " " + message)


def defineArguments(availableTests):
    parser = argparse.ArgumentParser(description="Test CF Functions")
    parser.add_argument(
        "--show", help="Show test outputs", dest="show_outputs", action="store_true"
    )
    parser.add_argument(
        "--all", help="Run All Tests", dest="run_all", action="store_true"
    )
    parser.add_argument("--dir", help="Output Directory for Tests")

    parser.add_argument(
        "--test", help="Choose Test to run", choices=list(availableTests.keys())
    )

    parser.set_defaults(show_outputs=False, run_all=False)

    return parser.parse_args()


def processInput(args, availableTests):
    if args.run_all:
        run_all_tests(args, availableTests)
        return
    if args.test:
        run_target_test(args, availableTests)
        return


def run_all_tests(args, availableTests):
    for test in availableTests:
        try:
            processOutputBefore(args, test)
            availableTests[test]()
            processOutputAfter()
            printSuccess(test + " Run Successfully")
        except Exception:
            processOutputAfter()
            printError(test + " Run with Error")


def run_target_test(args, availableTests):
    try:
        processOutputBefore(args, args.test)
        availableTests[args.test]()
        processOutputAfter()
        printSuccess(args.test + " Run Successfully")
    except Exception:
        processOutputAfter()
        printError(args.test + " Run with Error")
