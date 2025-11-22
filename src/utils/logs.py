import logging
import logging.config

def setup_logging():
    """Configure le logging pour toute l'application."""
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': '/tmp/app.log',
                'maxBytes': 1024 * 1024 * 5,
                'backupCount': 5,
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default', 'file'],
                'level': 'INFO',
                'propagate': True
            },
            'uvicorn': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False
            },
            # Ajoutez d'autres loggers pour vos modules spécifiques
            'rag_pipeline': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)

# Initialisation du logger pour ce module
logger = logging.getLogger(__name__)