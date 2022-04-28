


from cases.exceptions.module_runtime_exception import ModuleRuntimeException


def run_runtime():
    try:
        a = 1/0
    except Exception as e:
        raise ModuleRuntimeException(
            code=1,                         # HELP MD
            type="runtime.py",              # HELP MD
            msg=e                           # HELP End User
            )