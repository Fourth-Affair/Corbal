alter table public.plan_answer_feedback
  drop constraint if exists plan_answer_feedback_feedback_check;

alter table public.plan_answer_feedback
  add constraint plan_answer_feedback_feedback_check
  check (feedback in ('helpful', 'incorrect', 'unclear', 'trusted', 'rejected'));
