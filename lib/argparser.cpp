#include "argparser.h"


ArgParser::ArgParser(int &argc, const char **argv)
{
    for (int i = 1; i < argc; ++i)
        _tokens.push_back(std::string(argv[i]));
}


bool ArgParser::hasOption(const std::string &option) const
{
    return std::find(_tokens.begin(), _tokens.end(), option) != _tokens.end();
}
