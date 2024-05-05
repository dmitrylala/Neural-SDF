#include <catch2/catch_test_macros.hpp>

#include "siren.h"


TEST_CASE( "siren init", "[siren]" ) {
    auto net = getSirenNetwork(2, 64, 10);

    REQUIRE( net->m_batch_size == 10 );
}
