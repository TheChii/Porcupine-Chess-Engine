use porcupine::types::{Board, Color, Move, MoveFlag, Piece, Square};
use porcupine::eval::porcupine_nnue::{Model, PorcupineEvaluator};
use std::sync::Arc;

fn main() {
    let mut board = Board::default();
    let model = Model::load_embedded();
    let mut eval = PorcupineEvaluator::new(model, &board);
    
    println!("Start: {}", eval.evaluate(0, Color::White));
    
    // Play e4
    let mv = Move::new(Square::E2, Square::E4, MoveFlag::Quiet);
    board.push(mv);
    eval.update_move(0, &board, mv);
    println!("After e4 (Black to move): {}", eval.evaluate(1, Color::Black));
    
    // Evaluate from scratch to compare
    let mut eval2 = PorcupineEvaluator::new(Model::load_embedded(), &board);
    println!("After e4 (Scratch): {}", eval2.evaluate(0, Color::Black));
}
