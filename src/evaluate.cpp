/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the side to move. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos) {
    Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}

bool Eval::use_smallnet(const Position& pos) { 
    return std::abs(simple_eval(pos)) > 962; 
}

// Dynamic coefficient calculation using weighted influence factors
// This prevents cumulative penalization and maintains evaluation stability
std::pair<int, int> get_dynamic_coeffs(const Position& pos, int psqt, int positional) {
    // Base coefficients
    const int psqt_base = 125;
    const int pos_base = 131;
    
    // Start with neutral influence
    float psqt_influence = 1.0f;
    float pos_influence = 1.0f;
    
    // Game phase influence
    int total_pieces = popcount(pos.pieces() ^ pos.pieces(PAWN));
    if (total_pieces >= 20) {
        // Opening: favor PSQT for tactics and development
        psqt_influence *= 1.032f;  // +3.2%
        pos_influence *= 0.985f;   // -1.5%
    } else if (total_pieces <= 10) {
        // Endgame: favor positional for precision
        psqt_influence *= 0.976f;  // -2.4%
        pos_influence *= 1.031f;   // +3.1%
    }
    
    // NNUE component disagreement influence
    int nnue_diff = std::abs(psqt - positional);
    if (nnue_diff > 200) {
        // When components disagree, favor the more conservative (smaller magnitude)
        if (std::abs(psqt) > std::abs(positional)) {
            psqt_influence *= 0.984f;   // -1.6%
            pos_influence *= 1.008f;    // +0.8%
        } else {
            psqt_influence *= 1.008f;   // +0.8%
            pos_influence *= 0.984f;    // -1.6%
        }
    }
    
    // Material imbalance influence
    int material_diff = std::abs(pos.non_pawn_material(WHITE) - pos.non_pawn_material(BLACK));
    if (material_diff > 300) {
        // Imbalanced positions: slightly favor PSQT for tactical evaluation
        psqt_influence *= 1.016f;   // +1.6%
        pos_influence *= 0.992f;    // -0.8%
    }
    
    // Rule50 influence - approaching draw, favor positional accuracy
    if (pos.rule50_count() > 40) {
        float rule50_factor = std::min(0.02f, (pos.rule50_count() - 40) * 0.002f);
        psqt_influence *= (1.0f - rule50_factor);
        pos_influence *= (1.0f + rule50_factor);
    }
    
    // Apply influences and clamp to safe bounds
    int psqt_coeff = std::clamp(int(psqt_base * psqt_influence), 118, 132);
    int pos_coeff = std::clamp(int(pos_base * pos_influence), 124, 138);
    
    return {psqt_coeff, pos_coeff};
}

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks&    networks,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());

    bool smallNet           = use_smallnet(pos);
    auto [psqt, positional] = smallNet ? networks.small.evaluate(pos, accumulators, &caches.small)
                                       : networks.big.evaluate(pos, accumulators, &caches.big);

    // Apply dynamic coefficient adjustment
    auto [psqt_coeff, pos_coeff] = get_dynamic_coeffs(pos, psqt, positional);
    Value nnue = (psqt_coeff * psqt + pos_coeff * positional) / 128;

    // Re-evaluate the position when higher eval accuracy is worth the time spent
    if (smallNet && (std::abs(nnue) < 236))
    {
        std::tie(psqt, positional) = networks.big.evaluate(pos, accumulators, &caches.big);
        // Recalculate coefficients for big network evaluation
        auto [big_psqt_coeff, big_pos_coeff] = get_dynamic_coeffs(pos, psqt, positional);
        nnue = (big_psqt_coeff * psqt + big_pos_coeff * positional) / 128;
        smallNet = false;
    }

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = std::abs(psqt - positional);
    optimism += optimism * nnueComplexity / 468;
    nnue -= nnue * nnueComplexity / 18000;

    int material = 535 * pos.count<PAWN>() + pos.non_pawn_material();
    int v        = (nnue * (77777 + material) + optimism * (7777 + material)) / 77777;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 212;

    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    Eval::NNUE::AccumulatorStack accumulators;
    auto                         caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(networks);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, networks, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    auto [psqt, positional] = networks.big.evaluate(pos, accumulators, &caches->big);
    auto [psqt_coeff, pos_coeff] = get_dynamic_coeffs(pos, psqt, positional);
    
    // Show dynamic coefficients and their deviation from base
    ss << "Dynamic coefficients   PSQT: " << psqt_coeff << " (" 
       << std::showpos << (psqt_coeff - 125) << "), Positional: " << pos_coeff 
       << " (" << (pos_coeff - 131) << ")" << std::noshowpos << '\n';
    
    Value v                 = psqt + positional;
    v                       = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(networks, pos, accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "Final evaluation       " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with weighted influence coefficients]";
    ss << "\n";

    return ss.str();
}

}  // namespace Stockfish
