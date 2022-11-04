from enum import Enum


PREFIX = '@sys.'


class BaseEntityType(Enum):
    """ The enumeration of supported base entity types """

    NUMBER = PREFIX + 'number'
    DATE = PREFIX + 'date'
