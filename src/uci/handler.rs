//! UCI command handler and main loop.

use super::parser::{parse_command, UciCommand};
use super::{format_move, parse_move, SearchParams, ENGINE_AUTHOR, ENGINE_NAME};
use crate::book::PolyglotBook;
use crate::eval::nnue;
use crate::search::{SearchLimits, Searcher};
use crate::types::Board;
use std::io::{self, BufRead, Write};

/// UCI protocol handler
pub struct UciHandler {
    /// Current board position
    board: Board,
    /// Search engine (None when searching)
    searcher: Option<Searcher>,
    /// Receiver for completed searcher
    search_rx: Option<std::sync::mpsc::Receiver<Searcher>>,
    /// Shared state
    shared: std::sync::Arc<crate::search::SharedState>,
    /// Opening book
    book: Option<PolyglotBook>,
    /// Use opening book
    use_own_book: bool,
    /// Path to opening book file
    book_path: String,
    /// Debug mode enabled
    debug: bool,
    /// Should the engine quit
    quit: bool,
    /// Move overhead in milliseconds (safety buffer for time control)
    move_overhead: u64,
}

impl Default for UciHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl UciHandler {
    pub fn new() -> Self {
        let mut searcher = Searcher::new();

        // Load embedded NNUE model (compiled into the binary)
        match nnue::load_embedded_model() {
            Ok(model) => {
                // Log to stderr instead of stdout to avoid breaking strict UCI protocol on startup
                eprintln!("info string NNUE loaded: HalfKP (40960->256x2->32->32->1)");
                searcher.set_nnue(Some(model));
            }
            Err(e) => {
                eprintln!("info string NNUE load failed: {:?}", e);
            }
        }

        // Load embedded Porcupine NNUE (custom model)
        let model = crate::eval::porcupine_nnue::Model::load_embedded();
        eprintln!("info string Porcupine LOD-NNUE v2.2 (Nano) loaded: 22528->32->32->16->1 (network.bin)");
        searcher.porcupine = Some(model);

        let shared = searcher.shared.clone();

        Self {
            board: Board::default(),
            searcher: Some(searcher),
            search_rx: None,
            shared,
            book: None,
            use_own_book: false, // Disabled by default (standard UCI behavior)
            book_path: String::new(), // No default path
            debug: false,
            quit: false,
            move_overhead: 10, // Default 10ms
        }
    }

    /// Run the UCI main loop (blocking)
    pub fn run(&mut self) {
        let stdin = io::stdin();
        let reader = stdin.lock();

        for line in reader.lines() {
            match line {
                Ok(input) => {
                    if self.debug {
                        eprintln!("< {}", input);
                    }
                    self.handle_input(&input);
                    if self.quit {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }

    /// Handle a single UCI command
    pub fn handle_input(&mut self, input: &str) {
        let cmd = parse_command(input);
        self.handle_command(cmd);
    }

    fn handle_command(&mut self, cmd: UciCommand) {
        match cmd {
            UciCommand::Uci => self.cmd_uci(),
            UciCommand::Debug(on) => self.cmd_debug(on),
            UciCommand::IsReady => self.cmd_isready(),
            UciCommand::SetOption { name, value } => self.cmd_setoption(&name, value.as_deref()),
            UciCommand::Register => {} // Ignore registration
            UciCommand::UciNewGame => self.cmd_ucinewgame(),
            UciCommand::Position { fen, moves } => self.cmd_position(fen.as_deref(), &moves),
            UciCommand::Go(params) => self.cmd_go(params),
            UciCommand::Stop => self.cmd_stop(),
            UciCommand::PonderHit => self.cmd_ponderhit(),
            UciCommand::Quit => self.cmd_quit(),
            UciCommand::Display => self.cmd_display(),
            UciCommand::Unknown(s) => {
                if s.trim() == "eval" {
                    self.cmd_eval();
                } else if self.debug {
                    eprintln!("Unknown command: {}", s);
                }
            }
        }
    }

    fn cmd_eval(&mut self) {
        self.wait_for_search();
        let searcher = self.searcher.as_ref().unwrap();
        let mut evaluator = crate::eval::SearchEvaluator::new(
            searcher.eval_method,
            searcher.nnue.as_ref(),
            searcher.porcupine.as_deref(),
            &self.board
        );
        let score = evaluator.evaluate(0, &self.board);
        println!("eval {} (method: {:?})", score, searcher.eval_method);
    }

    /// Send output to GUI
    fn send(&self, msg: &str) {
        println!("{}", msg);
        io::stdout().flush().ok();
    }

    // === UCI Commands ===

    fn cmd_uci(&self) {
        self.send(&format!("id name {}", ENGINE_NAME));
        self.send(&format!("id author {}", ENGINE_AUTHOR));

        // Send options
        self.send("option name Hash type spin default 16 min 1 max 16384");
        self.send("option name Threads type spin default 1 min 1 max 64");
        self.send("option name MoveOverhead type spin default 10 min 0 max 5000");
        self.send("option name OwnBook type check default false");
        self.send("option name BookPath type string default <empty>");
        // Removed EvalMethod option

        // Send HCE tuning options
        self.send("option name pawn_mg type spin default 100 min -1000 max 1000");
        self.send("option name pawn_eg type spin default 120 min -1000 max 1000");
        self.send("option name knight_mg type spin default 320 min -1000 max 1000");
        self.send("option name knight_eg type spin default 300 min -1000 max 1000");
        self.send("option name bishop_mg type spin default 330 min -1000 max 1000");
        self.send("option name bishop_eg type spin default 320 min -1000 max 1000");
        self.send("option name rook_mg type spin default 500 min -2000 max 2000");
        self.send("option name rook_eg type spin default 550 min -2000 max 2000");
        self.send("option name queen_mg type spin default 950 min -5000 max 5000");
        self.send("option name queen_eg type spin default 1000 min -5000 max 5000");
        self.send("option name bishoppair_mg type spin default 35 min -500 max 500");
        self.send("option name bishoppair_eg type spin default 50 min -500 max 500");
        
        for i in 2..=7 {
            self.send(&format!("option name passed_rank{}_mg type spin default 0 min -1000 max 1000", i));
            self.send(&format!("option name passed_rank{}_eg type spin default 0 min -1000 max 1000", i));
        }

        self.send("uciok");
    }

    fn cmd_debug(&mut self, on: bool) {
        self.debug = on;
    }

    fn cmd_isready(&self) {
        self.send("readyok");
    }

    fn cmd_setoption(&mut self, name: &str, value: Option<&str>) {
        match name.to_lowercase().as_str() {
            "hash" => {
                if let Some(v) = value {
                    if let Ok(mb) = v.parse::<usize>() {
                        self.wait_for_search();
                        self.searcher.as_mut().unwrap().set_hash_size(mb);
                        self.shared = self.searcher.as_ref().unwrap().shared.clone();
                        if self.debug {
                            eprintln!("Hash set to {} MB", mb);
                        }
                    }
                }
            }
            "threads" => {
                if let Some(v) = value {
                    if let Ok(n) = v.parse::<usize>() {
                        self.wait_for_search();
                        self.searcher.as_mut().unwrap().set_threads(n);
                    }
                }
            }
            "moveoverhead" => {
                if let Some(v) = value {
                    if let Ok(ms) = v.parse::<u64>() {
                        self.move_overhead = ms.min(5000);
                    }
                }
            }
            "ownbook" => {
                if let Some(v) = value {
                    self.use_own_book = v.to_lowercase() == "true";
                    if self.debug {
                        eprintln!("OwnBook set to: {}", self.use_own_book);
                    }

                    // If enabling OwnBook and we have a book path, load the book
                    if self.use_own_book && !self.book_path.is_empty() {
                        match PolyglotBook::load(&self.book_path) {
                            Ok(b) => {
                                eprintln!(
                                    "info string Opening book loaded: {} ({} entries)",
                                    b.desc,
                                    b.len()
                                );
                                self.book = Some(b);
                            }
                            Err(e) => {
                                eprintln!(
                                    "info string Failed to load book {}: {:?}",
                                    self.book_path, e
                                );
                                self.book = None;
                            }
                        }
                    } else if !self.use_own_book {
                        // If disabling OwnBook, unload the book
                        if self.book.is_some() {
                            if self.debug {
                                eprintln!("OwnBook disabled, unloading book");
                            }
                            self.book = None;
                        }
                    }
                }
            }
            "bookpath" => {
                if let Some(v) = value {
                    self.book_path = v.to_string();
                    // Only load the book if OwnBook is enabled
                    if self.use_own_book {
                        match PolyglotBook::load(&self.book_path) {
                            Ok(b) => {
                                eprintln!(
                                    "info string Opening book loaded: {} ({} entries)",
                                    b.desc,
                                    b.len()
                                );
                                self.book = Some(b);
                            }
                            Err(e) => {
                                eprintln!(
                                    "info string Failed to load book {}: {:?}",
                                    self.book_path, e
                                );
                                self.book = None;
                            }
                        }
                    } else {
                        if self.debug {
                            eprintln!("BookPath set but OwnBook is disabled, not loading book");
                        }
                    }
                }
            }

            _ => {
                // Try updating tuning parameters
                if let Some(v) = value {
                    if let Ok(val) = v.parse::<i32>() {
                        if crate::eval::hce::update_param(name, val) {
                            if self.debug {
                                eprintln!("Tuning parameter {} set to {}", name, val);
                            }
                            return;
                        }
                    }
                }

                if self.debug {
                    eprintln!("Unknown option: {}", name);
                }
            }
        }
    }

    fn cmd_ucinewgame(&mut self) {
        self.wait_for_search();
        let mut searcher = self.searcher.take().unwrap();
        // Preserve NNUE models before resetting
        let nnue_model = searcher.nnue.take();
        let porcupine_model = searcher.porcupine.take();
        let eval_method = searcher.eval_method;
        let size_mb = searcher.shared.tt.size_mb();
        let threads = searcher.threads();

        self.board = Board::default();
        let mut new_searcher = Searcher::new();

        // Restore NNUE models
        new_searcher.nnue = nnue_model;
        new_searcher.porcupine = porcupine_model;
        new_searcher.eval_method = eval_method;
        new_searcher.set_hash_size(size_mb);
        new_searcher.set_threads(threads);

        self.shared = new_searcher.shared.clone();
        self.searcher = Some(new_searcher);
    }

    fn cmd_position(&mut self, fen: Option<&str>, moves: &[String]) {
        self.wait_for_search();
        // Set up the position
        self.board = match fen {
            Some(f) => Board::from_fen(f).unwrap_or_default(),
            None => Board::default(),
        };

        // Track position hashes for repetition detection
        let mut history: Vec<u64> = Vec::with_capacity(moves.len() + 1);
        history.push(self.board.hash());

        // Apply moves
        for move_str in moves {
            if let Some(m) = parse_move(&self.board, move_str) {
                self.board = self.board.make_move_new(m);
                history.push(self.board.hash());
            } else if self.debug {
                eprintln!("Invalid move: {}", move_str);
            }
        }

        // Store history in searcher for repetition detection
        self.searcher
            .as_mut()
            .unwrap()
            .set_position_with_history(self.board.clone(), history);
    }

    fn cmd_go(&mut self, params: SearchParams) {
        self.wait_for_search();

        // Try opening book first (unless infinite or analysis mode)
        if self.use_own_book && !params.infinite && params.searchmoves.is_empty() {
            if let Some(ref book) = self.book {
                if let Some(book_move) = book.probe_move(&self.board) {
                    self.send(&"info string book move".to_string());
                    self.send(&format!("bestmove {}", format_move(book_move)));
                    return;
                }
            }
        }

        // Set up search limits with move overhead
        let limits = SearchLimits::from_params(&params).with_move_overhead(self.move_overhead);

        let mut searcher = self.searcher.take().unwrap();
        // Position and history already set by cmd_position — don't overwrite

        let (tx, rx) = std::sync::mpsc::channel();
        self.search_rx = Some(rx);

        std::thread::spawn(move || {
            let result = searcher.search(limits);

            // Send best move
            match result.best_move {
                Some(m) => println!("bestmove {}", format_move(m)),
                None => println!("bestmove 0000"),
            }
            io::stdout().flush().ok();

            // Send searcher back
            let _ = tx.send(searcher);
        });
    }

    fn wait_for_search(&mut self) {
        if let Some(rx) = self.search_rx.take() {
            if let Ok(searcher) = rx.recv() {
                self.searcher = Some(searcher);
            }
        }
    }

    fn cmd_stop(&mut self) {
        self.shared
            .stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn cmd_ponderhit(&mut self) {
        self.shared
            .ponderhit
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn cmd_quit(&mut self) {
        self.shared
            .stop
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.quit = true;
    }

    fn cmd_display(&self) {
        // Non-standard debug command to display the board
        eprintln!("{:?}", self.board);
        eprintln!("FEN: {}", self.board.to_fen());
        eprintln!("Side to move: {:?}", self.board.turn());
    }
}

