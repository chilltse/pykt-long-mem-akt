import json
import pandas as pd
from .utils import sta_infos, write_txt

# KC: prev_frets | prev_strings | frets | strings
# 同一条 exercise 内首个有效帧 prev_frets / prev_strings 为 inf

KEYS = ["user_id", "sequence_id"]

# =============================================================================
# yousician JSON 预处理
# -----------------------------------------------------------------------------
# 1) 只保留 play_mode == "play" 的样本
# 2) 按 user_id 分组，同一用户内按 days_since_signup / session / part 排序后拼接
# 3) pitches 先转成 MIDI pitch list：首项为 base，后续项逐步累加
# 4) frets = midi_pitch - open_string_midi
# 5) question_id = prev_frets | prev_strings | frets | strings
# 6) KC（sequence_id）= prev_frets | prev_strings | frets | strings
# 7) response = reject_reason：0→1，非0→0
# =============================================================================


OPEN_MIDI_BY_STRING = {
    0: 69,
    1: 64,
    2: 60,
    3: 67,
}


def sanitize_field(x, sep="^"):
    """将字段转成字符串，并把逗号替换为 sep，避免和 write_txt 的逗号分隔冲突。"""
    s = str(x)
    s = s.replace(",", sep)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")
    return s


def _to_int_list(x):
    """将单值/列表/字符串统一解析成 int list。兼容 '[1,2]'、'1^2'、'1,2'。"""

    def _parse_int_tokens(tokens):
        out = []
        for v in tokens:
            sv = str(v).strip()
            if not sv:
                continue
            low = sv.lower()
            if low in {"none", "null", "nan"}:
                continue
            try:
                out.append(int(sv))
            except (TypeError, ValueError):
                try:
                    out.append(int(float(sv)))
                except (TypeError, ValueError):
                    continue
        return out

    if isinstance(x, list):
        return _parse_int_tokens(x)

    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    s = s.replace(",", "^")
    if not s:
        return []

    return _parse_int_tokens(s.split("^"))


def _cumulative_pitch_list(x):
    """
    Yousician pitches 编码：
    首项是 base MIDI pitch，后续项是相对前一个 pitch 的增量。
    
    例如:
        [60, 4, 3] -> [60, 64, 67]
    """
    vals = _to_int_list(x)
    if not vals:
        return []

    out = [vals[0]]
    for d in vals[1:]:
        out.append(out[-1] + d)

    return out


def _list_to_field(vals):
    """将 int list 转成稳定字段字符串，例如 [3, 0, 0, 0] -> '3^0^0^0'。"""
    return "^".join(str(int(v)) for v in vals)


def _fret4_list_from_pitch_and_string(pitch_x, string_x):
    """
    将 pitches + strings 映射为固定 4 维 fret 表示，顺序对应 string 0,1,2,3。

    未出现的弦补 0。
    注意：0 既可能表示空弦，也可能表示未出现；
    但因为 KC 里同时保留 strings 字段，所以可以区分当前实际弹了哪些弦。
    """
    mids = _cumulative_pitch_list(pitch_x)
    strs = _to_int_list(string_x)

    if len(mids) != len(strs):
        raise ValueError(f"音高数与弦数不一致: {mids!r} vs {strs!r}")

    fret4 = [0, 0, 0, 0]

    for m, s in zip(mids, strs):
        if s not in OPEN_MIDI_BY_STRING:
            raise ValueError(f"非法弦号 {s}，应为 0-3")

        fret4[s] = int(m) - OPEN_MIDI_BY_STRING[s]

    return fret4


def _strings_field(string_x):
    """
    将 strings 转成稳定字段。
    这里使用 sorted(set(...))，避免同一组弦因为顺序不同变成不同 KC。
    """
    strs = _to_int_list(string_x)
    strs = sorted(set(strs))
    return _list_to_field(strs)


def _events_to_event_rows(events_data_str, include_event_index=False):
    """
    从 events_data 解析 (question_id, kc, response)。
    include_event_index=True 时额外在每行开头返回过滤前的原始事件索引。

    prev_frets / prev_strings 在每个 exercise(record) 内计算。
    如果某一帧无法转 fret，会跳过该帧；
    下一帧的 prev 会使用上一条保留下来的有效帧。
    """
    data = json.loads(events_data_str)
    inner = data.get("data", data)

    duration = inner["duration"]
    pitches = inner["pitches"]
    strings = inner["strings"]
    reject_reason = inner["reject_reason"]

    n = len(pitches)
    if not (n == len(duration) == len(strings) == len(reject_reason)):
        raise ValueError(
            f"长度不一致: pitches={n}, duration={len(duration)}, "
            f"strings={len(strings)}, reject_reason={len(reject_reason)}"
        )

    out = []

    prev_frets_s = "inf"
    prev_strings_s = "inf"

    for i in range(n):
        try:
            curr_fret4 = _fret4_list_from_pitch_and_string(pitches[i], strings[i])
            curr_frets_s = _list_to_field(curr_fret4)
            curr_strings_s = _strings_field(strings[i])
        except ValueError:
            continue

        if curr_strings_s == "":
            continue

        qid = f"{prev_frets_s}|{prev_strings_s}|{curr_frets_s}|{curr_strings_s}"
        kc = qid

        resp = 1 if reject_reason[i] == 0 else 0
        row = (qid, kc, resp)
        out.append((i,) + row if include_event_index else row)

        prev_frets_s = curr_frets_s
        prev_strings_s = curr_strings_s

    return out


def read_data_from_json(read_file, write_file):
    stares = []

    with open(read_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    records = [r for r in raw_list if r.get("play_mode") == "play"]
    if not records:
        raise ValueError("No records with play_mode=='play'")

    rows = []
    for r in records:
        uid = r["user_id"]
        days = r["days_since_signup"]
        part = r.get("exercise_part_index", 0)
        sess = r.get("session_index", 0)

        try:
            event_list = _events_to_event_rows(r["events_data"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

        for qid, kc, resp in event_list:
            rows.append(
                {
                    "user_id": uid,
                    "days_since_signup": days,
                    "exercise_part_index": part,
                    "session_index": sess,
                    "question_id": qid,
                    "sequence_id": kc,
                    "correct": resp,
                }
            )

    df = pd.DataFrame(rows)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"after filter play_mode=play, interaction num: {ins}, user num: {us}, "
        f"question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, "
        f"avg(c) per q: {avgcq}, na: {na}"
    )

    user_inters = []
    for user, grp in df.groupby("user_id", sort=False):
        tmp = grp.sort_values(
            by=["days_since_signup", "session_index", "exercise_part_index"]
        )

        seq_problems = tmp["question_id"].astype(str).tolist()
        seq_skills = tmp["sequence_id"].astype(str).tolist()
        seq_ans = tmp["correct"].astype(str).tolist()

        seq_len = len(seq_problems)
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        user_inters.append(
            [
                [str(user), str(seq_len)],
                seq_problems,
                seq_skills,
                seq_ans,
                seq_start_time,
                seq_response_cost,
            ]
        )

    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return