use ferrum_movegen::board::Board;
fn main() {
    let board = Board::from_fen("k7/8/8/8/3q4/2P5/8/RNBQKBNR w KQkq - 0 1").unwrap();
    let moves = board.generate_moves();
    for m in moves.iter() {
        let nb = board.make_move_new(*m);
    }
    println!("Done without panic!");
}
