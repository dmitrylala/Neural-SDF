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


int ArgParser::get_n_threads() const
{
    int n_threads = 1;
    if (hasOption("--n_threads")) {
        int got_threads = getOptionValue<int>("--n_threads");
        n_threads = std::max(got_threads, 1);
    }
    return n_threads;
}
