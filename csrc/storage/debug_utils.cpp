// -------------------------------------
// Debugging and timing macros
// -------------------------------------

#define DEBUG_PRINT(msg)                                                     \
    do {                                                                     \
        const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG");            \
        if (env && std::string(env) != "0")                                 \
            std::cout << "[DEBUG] " << msg << std::endl;                     \
    } while (0)

// Timing macro - measures only if STORAGE_CONNECTOR_DEBUG is not "0"
#define TIME_EXPR(label, expr, info_str) ([&]() {                                  \
    const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG");                      \
    if (!(env && std::string(env) == "1")) {                                       \
        return (expr);                                                             \
    }                                                                              \
    auto __t0 = std::chrono::high_resolution_clock::now();                         \
    auto __ret = (expr);                                                           \
    auto __t1 = std::chrono::high_resolution_clock::now();                         \
    double __ms = std::chrono::duration<double, std::milli>(__t1 - __t0).count();  \
    std::cout << "[DEBUG][TIME] " << label << " took " << __ms << " ms | "         \
              << info_str << std::endl;                                            \
    return __ret;                                                                  \
})()
