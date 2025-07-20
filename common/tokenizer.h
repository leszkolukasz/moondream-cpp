#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <stdexcept>
#include <regex>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

namespace moondream {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& config_path) {
        std::ifstream input(config_path);
        if (!input.is_open()) {
            throw std::runtime_error("Unable to open tokenizer config file.");
        }
        nlohmann::json j;
        input >> j;
        parse_config(j);
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<std::string> words = simple_split(text);
        std::vector<std::string> tokens;

        for (const auto& word : words) {
            auto pieces = byte_pair_encode(word);
            tokens.insert(tokens.end(), pieces.begin(), pieces.end());
        }

        std::vector<int> token_ids;
        for (const auto& token : tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                token_ids.push_back(it->second);
            } else if (unk_token.has_value() && vocab.count(unk_token.value())) {
                token_ids.push_back(vocab.at(unk_token.value()));
            } else {
                throw std::runtime_error("Token not in vocab and no unk_token defined: " + token);
            }
        }
        return token_ids;
    }

    std::string decode(const std::vector<int>& token_ids) const {
        std::ostringstream result;
        for (const auto& id : token_ids) {
            auto it = inv_vocab.find(id);
            result << (it != inv_vocab.end() ? it->second : unk_token.value_or("UNK"));
        }
        std::string output = result.str();
        size_t pos = 0;
        while ((pos = output.find("Ġ", pos)) != std::string::npos) {
            output.replace(pos, 3, " ");
            ++pos;
        }
        return output;
    }

private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    std::map<std::string, std::string> merges;
    std::optional<std::string> unk_token;

    void parse_config(const nlohmann::json& j) {
        const auto& model = j.at("model");
        if (model.at("unk_token").is_null()) {
            unk_token = std::nullopt;
        } else {
            unk_token = model.at("unk_token").get<std::string>();
        }

        for (const auto& [key, val] : model.at("vocab").items()) {
            vocab[key] = val;
            inv_vocab[val] = key;
        }

        for (const auto& pair : model.at("merges")) {
            std::string a = pair[0];
            std::string b = pair[1];
            std::string merged = a + b;
            merges[merged] = a + " " + b;
        }
    }

    std::vector<std::string> simple_split(const std::string& text) const {
        std::vector<std::string> tokens;
        static const std::regex pattern(R"([\w]+|[^\s\w])");
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
        auto words_end = std::sregex_iterator();

        for (auto it = words_begin; it != words_end; ++it) {
            tokens.push_back(it->str());
        }
        return tokens;
    }

    std::vector<std::string> byte_pair_encode(const std::string& word) const {
        std::vector<std::string> tokens;
        for (char c : word) {
            tokens.emplace_back(1, c);
        }

        while (true) {
            bool merged = false;
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                std::string pair = tokens[i] + tokens[i + 1];
                auto it = merges.find(pair);
                if (it != merges.end()) {
                    tokens[i] = pair;
                    tokens.erase(tokens.begin() + i + 1);
                    merged = true;
                    break;
                }
            }
            if (!merged) break;
        }
        return tokens;
    }
};

}  // namespace moondream
