#!/usr/bin/env python3
import click


class NotRequiredIf(click.Option):
    def __init__(self, *args, **kwargs):
        self.not_required_if = kwargs.pop("not_required_if")
        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs["help"] = (
            kwargs.get("help", "")
            + f" NOTE: This argument is mutually exclusive with {self.not_required_if}"
        ).strip()
        super(NotRequiredIf, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        opt_present = self.name in opts
        not_required_present = self.not_required_if in opts

        if opt_present:
            if not_required_present:
                raise click.UsageError(
                    f"Illegal usage: `{self.name}` is mutually"
                    + f" exclusive with `{self.not_required_if}`"
                )
            else:
                self.prompt = None

        return super(NotRequiredIf, self).handle_parse_result(ctx, opts, args)
