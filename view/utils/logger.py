import logging, sys, os
def get_logger(name="service"):
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    lg.addHandler(h)
    fh = logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "service.log"))
    fh.setFormatter(h.formatter); lg.addHandler(fh)
    return lg
