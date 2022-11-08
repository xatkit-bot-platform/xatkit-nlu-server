from enum import Enum


PREFIX = '@sys.'


class BaseEntityType:
    """ The enumeration of supported base entity types """

    NUMBER = PREFIX + 'number'
    DATE = PREFIX + 'date'
    ANY = PREFIX + 'any'


ordered_base_entities = [
    BaseEntityType.DATE,
    BaseEntityType.NUMBER,
    BaseEntityType.ANY
]