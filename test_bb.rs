use movegen::Board;
use movegen::Piece;
use movegen::Color;

fn main() {
    let board = Board::startpos();
    println!("Testing piece_bb and color_bb for starting position:");
    
    // Test white pawns
    let white_pawns = board.piece_bb(Piece::Pawn) & board.color_bb(Color::White);
    println!("White pawns: {} (should be 8)", white_pawns.count());
    
    // Test black pawns
    let black_pawns = board.piece_bb(Piece::Pawn) & board.color_bb(Color::Black);
    println!("Black pawns: {} (should be 8)", black_pawns.count());
    
    // Test all pieces
    for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        let w = (board.piece_bb(piece) & board.color_bb(Color::White)).count();
        let b = (board.piece_bb(piece) & board.color_bb(Color::Black)).count();
        println!("{:?} W:{} B:{}", piece, w, b);
    }
}
