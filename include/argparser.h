#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <optional>


class ArgParser
{
public:
    ArgParser(int &argc, const char **argv);

    template <typename T>
    T getOptionValue(const std::string &option, std::optional<T> defaultValue=std::nullopt) const
    {
        std::vector<std::string>::const_iterator itr;

        itr = std::find(_tokens.begin(), _tokens.end(), option);
        if (itr == _tokens.end() || ++itr == _tokens.end()) {
            if (defaultValue.has_value()) {
                return defaultValue.value();
            }
            std::stringstream ss;
            ss << "Option " << option << " doesn't have any value";
            throw std::runtime_error(ss.str());
        }

        return from_string<T>(*itr);
    }

    bool hasOption(const std::string &option) const;

private:
    std::vector<std::string> _tokens;

    template <typename T>
    T from_string(const std::string &str) const
    {
        if constexpr(std::is_same_v<T, int>) {
            return std::stoi(str);
        } else if constexpr(std::is_same_v<T, float>) {
            return std::stof(str);
        } else
            return str;
    }
};
