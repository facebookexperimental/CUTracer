# Blind prompt

Fill in `{{PACKET}}` with the full text of one case's `packet.md`, then give the result to the agent under test. Give it nothing else from this directory.

```text
You are drafting a post for the CUTracer Workplace group from the completed investigation packet below. The group feed shows only the post title, and readers search the group by tag.

Write exactly three things:

1. The post title, on one line.
2. The Tags line for the post.
3. A one-sentence TL;DR.

Use only facts in the packet. Keep the packet's conclusion at the strength the packet states.

<packet>
{{PACKET}}
</packet>
```
