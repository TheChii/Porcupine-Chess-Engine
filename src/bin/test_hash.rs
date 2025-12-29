fn main() {
    let board = movegen::Board::startpos();
    println!("Starting position hash: 0x{:016X}", board.hash());
    
    let moves = board.generate_moves();
    println!("Generated {} moves", moves.len());
    
    for m in moves.iter().take(5) {
        let new_board = board.make_move_new(m);
        println!("After {}: hash = 0x{:016X}", m, new_board.hash());
    }
}
