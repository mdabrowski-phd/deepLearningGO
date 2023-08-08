#%% Listing 8.12 Python implementation of a GTP command
class Command:

    def __init__(self, sequence, name, args):
        self.sequence = sequence
        self.name = name
        self.args = tuple(args)

    def __eq__(self, other):
        return self.sequence == other.sequence and \
            self.name == other.name and \
            self.args == other.args

    def __repr__(self):
        return 'Command(%r, %r, %r)' % (self.sequence, self.name, self.args)

    def __str__(self):
        return repr(self)

#%% Listing 8.13 Partisn a GTP Command from plain text
def parse(command_string):
    pieces = command_string.split()
    try:
        sequence = int(pieces[0]) # <1>
        pieces = pieces[1:]
    except ValueError:  # <2>
        sequence = None
    name, args = pieces[0], pieces[1:]
    return Command(sequence, name, args)