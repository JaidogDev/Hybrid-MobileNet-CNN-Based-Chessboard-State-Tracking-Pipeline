import chess

MAP = {
 "WP":"P","WN":"N","WB":"B","WR":"R","WQ":"Q","WK":"K",
 "BP":"p","BN":"n","BB":"b","BR":"r","BQ":"q","BK":"k"
}

def labels_to_board(labels8x8):
    b = chess.Board.empty()
    for r in range(8):
        for c in range(8):
            lab = labels8x8[r][c]
            if lab and lab!="Empty":
                sq = chess.square(c, 7-r)   # map (r,c) → chess square
                b.set_piece_at(sq, chess.Piece.from_symbol(MAP[lab]))
    return b

def diff_to_move(prev_labels, now_labels):
    prev_b = labels_to_board(prev_labels)
    now_b  = labels_to_board(now_labels)

    from_sq = to_sq = None
    for sq in chess.SQUARES:
        if prev_b.piece_at(sq) != now_b.piece_at(sq):
            if prev_b.piece_at(sq) and not now_b.piece_at(sq): from_sq = sq
            if (not prev_b.piece_at(sq)) and now_b.piece_at(sq): to_sq = sq

    if from_sq is None or to_sq is None:
        return None
    # try normal + promotions
    m = chess.Move(from_sq, to_sq)
    if m in prev_b.legal_moves: return m
    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        pm = chess.Move(from_sq, to_sq, promotion=promo)
        if pm in prev_b.legal_moves: return pm
    return None

def san_list_to_pgn(sans):
    out = []; i=0; move_no=1
    while i < len(sans):
        if i+1 < len(sans):
            out.append(f"{move_no}. {sans[i]} {sans[i+1]}")
            i += 2
        else:
            out.append(f"{move_no}. {sans[i]}")
            i += 1
        move_no += 1
    return " ".join(out)
