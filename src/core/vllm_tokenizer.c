/**
 * vllm_tokenizer.c - Simple Word-Level Tokenizer
 *
 * Built-in vocabulary of ~2000 common English words, subwords,
 * and special tokens. Uses prefix-based greedy matching.
 */

#include "vllm_superpos.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* Built-in vocabulary: common words, subwords, and special tokens.
 * Index 0 = <unk>, 1 = <s>, 2 = </s>, 3 = <pad> */
/* Exported for tokenizer_decode_direct() */
const char *g_token_strings[913];
const int g_token_count = 913;

static const char *builtin_vocab[] = {
    "<unk>", "<s>", "</s>", "<pad>",
    /* Common words */
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "shall",
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "here", "there", "where",
    "what", "which", "who", "whom", "whose", "when", "why", "how",
    "not", "no", "nor", "and", "or", "but", "if", "then", "else",
    "so", "as", "at", "by", "for", "from", "in", "into", "of", "on",
    "to", "with", "without", "about", "above", "after", "against",
    "along", "among", "around", "before", "behind", "between",
    "beyond", "during", "except", "inside", "near", "off", "over",
    "since", "through", "under", "until", "up", "upon",
    "all", "any", "both", "each", "every", "few", "many", "more",
    "most", "much", "other", "some", "such", "only", "own", "same",
    "very", "just", "too", "also", "even", "still", "yet", "now",
    "again", "always", "never", "often", "sometimes", "usually",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "last", "next", "new", "old", "good",
    "bad", "big", "small", "large", "little", "high", "low", "long",
    "short", "great", "right", "left", "top", "bottom",
    "time", "year", "day", "week", "month", "hour", "minute",
    "world", "life", "hand", "part", "place", "case", "point",
    "group", "number", "people", "person", "man", "woman", "child",
    "children", "family", "friend", "name", "word", "thing",
    "way", "end", "fact", "kind", "form", "head", "side", "home",
    "water", "food", "land", "air", "fire", "light", "night",
    "work", "book", "story", "city", "country", "school",
    "state", "money", "power", "order", "question", "problem",
    "idea", "example", "change", "line", "reason", "interest",
    "make", "made", "take", "took", "come", "came", "see", "saw",
    "know", "knew", "think", "thought", "say", "said", "tell", "told",
    "get", "got", "go", "went", "give", "gave", "use", "used",
    "find", "found", "want", "like", "need", "let", "call", "help",
    "try", "ask", "show", "turn", "start", "keep", "play", "run",
    "move", "live", "mean", "read", "write", "learn", "understand",
    "believe", "happen", "bring", "hold", "put", "set", "become",
    "begin", "leave", "seem", "look", "feel", "hear", "listen",
    "speak", "talk", "walk", "stand", "sit", "eat", "drink",
    "open", "close", "build", "create", "develop", "design",
    "program", "code", "data", "system", "computer", "network",
    "server", "client", "model", "algorithm", "function", "result",
    "process", "method", "value", "type", "input", "output",
    "error", "test", "check", "file", "memory", "user", "language",
    "class", "object", "library", "tool", "project", "version",
    "science", "math", "physics", "energy", "space", "matter",
    "force", "field", "theory", "rule", "law", "history",
    "body", "mind", "heart", "eye", "hand", "face", "voice",
    "color", "red", "blue", "green", "black", "white",
    "love", "war", "peace", "art", "music", "film", "game",
    "team", "player", "level", "score", "role", "skill",
    "plan", "action", "figure", "cost", "price", "rate",
    "market", "company", "business", "service", "product",
    "customer", "report", "issue", "center", "area",
    "health", "death", "mind", "thought", "feeling", "sense",
    "order", "nature", "church", "court", "police", "army",
    "president", "government", "party", "job", "training",
    "community", "society", "culture", "education", "research",
    "source", "resource", "material", "paper", "image",
    "video", "text", "message", "email", "website", "page",
    "section", "list", "term", "note", "record", "table",
    "card", "board", "room", "door", "window", "wall", "floor",
    "tree", "plant", "animal", "bird", "fish", "dog", "cat",
    "car", "train", "plane", "ship", "road", "river", "sea",
    "mountain", "forest", "sky", "sun", "moon", "star", "cloud",
    "rain", "snow", "wind", "storm", "earth", "gold", "silver",
    "stone", "wood", "steel", "glass", "paper", "metal",
    "strong", "weak", "hard", "soft", "fast", "slow", "early",
    "late", "young", "happy", "sad", "true", "false", "real",
    "free", "full", "easy", "hard", "clear", "dark", "cold", "hot",
    "sure", "ready", "able", "simple", "complex", "single",
    "double", "possible", "important", "special", "general",
    "public", "private", "local", "national", "international",
    "political", "social", "economic", "technical", "natural",
    "human", "physical", "mental", "personal", "professional",
    "original", "final", "total", "average", "standard", "official",
    /* Common subword pieces */
    "ing", "ed", "ly", "tion", "sion", "ment", "ness", "able",
    "ible", "ful", "less", "ous", "ive", "al", "ic", "er", "est",
    "re", "un", "pre", "pro", "con", "dis", "ex", "sub", "inter",
    "trans", "micro", "multi", "non", "over", "semi", "super",
    "th", "ch", "sh", "ph", "gh", "wh", "qu", "ck", "ng", "nk",
    /* Punctuation & symbols as tokens */
    ".", ",", "!", "?", ";", ":", "\"", "'", "(", ")", "[", "]",
    "{", "}", "-", "-", "/", "\\", "@", "#", "$", "%", "^", "&",
    "*", "+", "=", "<", ">", "|", "~", "`",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    /* Additional words to fill vocabulary */
    "consider", "include", "continue", "provide", "require",
    "produce", "support", "involve", "receive", "accept",
    "expect", "decide", "allow", "return", "follow", "remain",
    "prepare", "achieve", "improve", "increase", "reduce",
    "apply", "compare", "explain", "describe", "suggest",
    "indicate", "represent", "determine", "establish", "consider",
    "approach", "practice", "principle", "concept", "context",
    "analysis", "strategy", "structure", "pattern", "feature",
    "aspect", "element", "factor", "purpose", "opportunity",
    "challenge", "solution", "benefit", "effect", "impact",
    "quality", "performance", "efficiency", "capacity", "growth",
    "progress", "success", "failure", "risk", "security",
    "protection", "response", "attention", "effort", "ability",
    "knowledge", "experience", "relationship", "connection",
    "tradition", "technology", "innovation", "application",
    "operation", "management", "organization", "development",
    "production", "construction", "communication", "information",
    "environment", "condition", "situation", "position", "direction",
    "population", "election", "campaign", "policy", "treatment",
    "discussion", "agreement", "argument", "conclusion",
    "decision", "statement", "evidence", "opinion", "difference",
    "advantage", "attention", "behavior", "movement", "activity",
    "industry", "economy", "trade", "investment", "employment",
    "education", "training", "medicine", "disease", "hospital",
    "accident", "crime", "violence", "conflict", "defense",
    "justice", "freedom", "democracy", "leadership", "authority",
    "member", "committee", "council", "agency", "department",
    "corporation", "institution", "association", "foundation",
    "property", "equipment", "instruction", "regulation",
    "requirement", "limitation", "explanation", "assumption",
    "interpretation", "implication", "recommendation",
    "characteristic", "responsibility", "possibility",
    /* Misc nouns */
    "chair", "table", "desk", "bed", "kitchen", "bathroom",
    "garden", "park", "street", "bridge", "building", "office",
    "store", "shop", "hotel", "restaurant", "airport", "station",
    "hospital", "church", "museum", "library", "theater",
    "teacher", "student", "doctor", "nurse", "lawyer", "judge",
    "officer", "soldier", "worker", "manager", "director",
    "writer", "artist", "musician", "singer", "actor", "athlete",
    /* Verbs */
    "open", "close", "push", "pull", "cut", "break", "drop",
    "rise", "fall", "grow", "die", "kill", "fight", "win", "lose",
    "draw", "paint", "sing", "dance", "cook", "clean", "wash",
    "drive", "fly", "swim", "jump", "throw", "catch", "hit",
    /* Adjectives */
    "beautiful", "ugly", "rich", "poor", "smart", "stupid",
    "brave", "afraid", "lucky", "lonely", "busy", "tired",
    "hungry", "thirsty", "angry", "excited", "bored", "proud",
    /* More subwords / suffixes */
    "ize", "ise", "ify", "ate", "ence", "ance", "ure", "age",
    "dom", "hood", "ship", "ward", "wise", "fold",
    /* Numbers extended */
    "hundred", "thousand", "million", "billion",
    /* Single character tokens for char-level fallback */
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z",
    /* Space token */
    " ",
    /* Capital letters */
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
};

#define BUILTIN_VOCAB_SIZE \
    (sizeof(builtin_vocab) / sizeof(builtin_vocab[0]))

void tokenizer_init(Tokenizer *tok) {
    int n = BUILTIN_VOCAB_SIZE;
    if (n > TINY_VOCAB_SIZE) n = TINY_VOCAB_SIZE;

    tok->vocab_size = n;
    tok->max_token_len = 0;
    tok->bos_id = 1;  /* <s> */
    tok->eos_id = 2;  /* </s> */
    tok->tokens = malloc(n * sizeof(char*));

    for (int i = 0; i < n; i++) {
        tok->tokens[i] = strdup(builtin_vocab[i]);
        g_token_strings[i] = tok->tokens[i];  /* export for direct decode */
        int len = (int)strlen(builtin_vocab[i]);
        if (len > tok->max_token_len) tok->max_token_len = len;
    }
}

void tokenizer_free(Tokenizer *tok) {
    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->tokens[i]);
    }
    free(tok->tokens);
    tok->tokens = NULL;
    tok->vocab_size = 0;
}

/**
 * Direct token decode: returns the string for a given token ID.
 * Used by weights_init for character-overlap embedding initialization.
 * Returns NULL if tok is NULL or id out of range.
 */
const char *tokenizer_decode_direct(int token_id) {
    /* Built-in vocabulary: only available after tokenizer_init() is called */
    extern const char *g_token_strings[];
    extern const int g_token_count;
    if (token_id < 0 || token_id >= g_token_count) return NULL;
    return g_token_strings[token_id];
}

/**
 * Find longest matching token at the start of text.
 * Returns token ID or -1 if no match.
 * Uses exact (case-sensitive) matching for BPE tokenizer compatibility.
 */
static int find_longest_match(Tokenizer *tok, const char *text, int text_len,
                               int *match_len) {
    int best_id = -1;
    int best_len = 0;

    for (int i = 0; i < tok->vocab_size; i++) {
        int tlen = (int)strlen(tok->tokens[i]);
        if (tlen > text_len) continue;

        /* Exact match */
        if (memcmp(tok->tokens[i], text, tlen) == 0) {
            if (tlen > best_len) {
                best_id = i;
                best_len = tlen;
            }
        }
    }
    *match_len = best_len;
    return best_id;
}

int tokenizer_encode(Tokenizer *tok, const char *text,
                      int *token_ids, int max_len) {
    int text_len = (int)strlen(text);
    int pos = 0;
    int num_tokens = 0;
    int is_first_word = 1;

    /* Do NOT prepend BOS automatically — caller controls BOS placement */

    while (pos < text_len && num_tokens < max_len) {
        int had_space = 0;
        int had_newline = 0;

        /* Handle newline characters explicitly — critical for chat templates */
        if (text[pos] == '\n') {
            /* Try to find '\n' as a token (common in LLaMA: token 13) */
            int ml;
            int tid = find_longest_match(tok, "\n", 1, &ml);
            if (tid >= 0 && ml > 0) {
                token_ids[num_tokens++] = tid;
            }
            pos++;
            is_first_word = 1;
            continue;
        }

        /* Skip other whitespace */
        while (pos < text_len && text[pos] != '\n' && isspace((unsigned char)text[pos])) {
            pos++;
            had_space = 1;
        }
        if (pos >= text_len) break;

        /* For BPE tokenizers (LLaMA), word continuations use Ġ (U+0120,
         * bytes 0xC4 0xA0) as the BPE space prefix. Try matching with
         * this prefix first (e.g., "Ġcapital" not "capital").
         * For the first word, use bare matching (no prefix). */
        int match_len = 0;
        int token_id = -1;
        const char *rem = text + pos;
        int rem_len = text_len - pos;

        if (!is_first_word) {
            /* Subsequent words: prepend Ġ (C4 A0) and try matching "Ġword" first */
            char spaced_buf[258]; /* 256 + 2 bytes for Ġ prefix */
            int buf_len = rem_len + 2 < 257 ? rem_len + 2 : 256;
            spaced_buf[0] = '\xC4';
            spaced_buf[1] = '\xA0';
            if (buf_len - 2 > 0) memcpy(spaced_buf + 2, rem, buf_len - 2);
            spaced_buf[buf_len] = '\0';
            token_id = find_longest_match(tok, spaced_buf, buf_len, &match_len);
            if (token_id >= 0 && match_len > 0) {
                pos += (match_len - 2);  /* -2 because we prepended 2-byte prefix */
                token_ids[num_tokens++] = token_id;
                is_first_word = 0;
                continue;
            }
        }

        /* Try matching without space prefix (first word or fallback) */
        token_id = find_longest_match(tok, rem, rem_len, &match_len);
        if (token_id >= 0 && match_len > 0) {
            token_ids[num_tokens++] = token_id;
            pos += match_len;
        } else {
            /* Character-level fallback */
            char c = text[pos];
            char s[2] = {c, '\0'};
            token_id = find_longest_match(tok, s, 1, &match_len);
            if (token_id >= 0) {
                token_ids[num_tokens++] = token_id;
            } else {
                token_ids[num_tokens++] = 0; /* <unk> */
            }
            pos++;
        }
        is_first_word = 0;
    }

    return num_tokens;
}

const char* tokenizer_decode(Tokenizer *tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    /* Don't return <unk> for readability */
    if (token_id == 0) return "";
    return tok->tokens[token_id];
}
