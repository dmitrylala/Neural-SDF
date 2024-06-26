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


std::tuple<int,int,int> ArgParser::get_network_setup() const
{
    int n_hidden_layers = getOptionValue<int>("--n_hidden");
    int hidden_size = getOptionValue<int>("--hidden_size");
    int batch_size = getOptionValue<int>("--batch_size");
    return std::tuple<int,int,int>{n_hidden_layers, hidden_size, batch_size};
}
