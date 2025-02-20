#pragma once

namespace erl::gaussian_process {

    extern bool initialized;

    /**
     * @brief Initialize the library.
     */
    bool
    Init();

    inline const static bool kAutoInitialized = Init();

}  // namespace erl::gaussian_process
