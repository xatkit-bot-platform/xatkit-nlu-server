PREFIX = '@sys.'


class BaseEntityType:
    """ The enumeration of supported base entity types """

    NUMBER = PREFIX + 'number'
    DATETIME = PREFIX + 'date-time'
    ANY = PREFIX + 'any'


ordered_base_entities = [
    BaseEntityType.DATETIME,
    BaseEntityType.NUMBER,
    BaseEntityType.ANY
]
